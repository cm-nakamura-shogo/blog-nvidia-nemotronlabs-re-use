"""mlx-speech の selective_scan を Metal カーネルで置き換える。

mlx-speech の実装は時間ステップごとの Python ループで、
deltaA を [B, d_inner, L, d_state] で一括に確保する。
44.1kHz 5 秒の入力では 1 本あたり約 7GB になり、16GB の Mac では現実的に動かない。

このカーネルは 1 スレッドが 1 つの (batch, channel) を担当し、
d_state 個の状態をレジスタに持ったまま L 方向へ逐次に進める。
計算式は mamba_ssm の selective_scan_ref と同じ（softplus の閾値 20 も torch に合わせる）。
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = """
    // u, delta, z, out は [B, L, d_inner]、B, C は [B, L, d_state] の並び。
    // 隣接スレッドが隣接 channel を読むので、L 方向の各ステップで読み書きがまとまる。
    uint gid = thread_position_in_grid.x;
    int Bt = u_shape[0];
    int L = u_shape[1];
    int Dn = u_shape[2];
    if (gid >= uint(Bt * Dn)) {
        return;
    }
    int b = gid / Dn;
    int d = gid % Dn;

    float a[N];
    float x[N];
    for (int n = 0; n < N; ++n) {
        a[n] = A[d * N + n];
        x[n] = 0.0f;
    }
    float d_skip = D[d];
    float d_bias = delta_bias[d];

    for (int t = 0; t < L; ++t) {
        size_t i = (size_t(b) * L + t) * Dn + d;
        size_t bc = (size_t(b) * L + t) * N;
        float dt = delta[i] + d_bias;
        // torch.nn.functional.softplus(beta=1, threshold=20)
        dt = dt > 20.0f ? dt : (dt < -15.0f ? metal::exp(dt) : metal::log(1.0f + metal::exp(dt)));
        float uu = u[i];
        float y = 0.0f;
        for (int n = 0; n < N; ++n) {
            x[n] = metal::exp(dt * a[n]) * x[n] + dt * B[bc + n] * uu;
            y += x[n] * C[bc + n];
        }
        y += uu * d_skip;
        float zz = z[i];
        y *= zz / (1.0f + metal::exp(-zz));
        out[i] = y;
    }
"""

_kernel = mx.fast.metal_kernel(
    name="reuse_selective_scan",
    input_names=["u", "delta", "A", "B", "C", "D", "z", "delta_bias"],
    output_names=["out"],
    source=_SOURCE,
)


def selective_scan(
    u: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array | None = None,
    z: mx.array | None = None,
    delta_bias: mx.array | None = None,
    delta_softplus: bool = True,
) -> mx.array:
    """mlx_speech.models.reuse.mamba.scan.selective_scan と同じ引数と戻り値。

    SEMamba が使う組み合わせ（D、z、delta_bias あり、softplus あり）だけに対応する。
    """
    if D is None or z is None or delta_bias is None or not delta_softplus:
        raise NotImplementedError("SEMamba の呼び出し形のみ対応")

    def f32_lmajor(t: mx.array) -> mx.array:
        # 呼び出し側は [B, L, C] を転置したビューを渡すので、戻すとほぼコピーなしで連続になる
        return mx.contiguous(mx.transpose(t, (0, 2, 1)).astype(mx.float32))

    def f32(t: mx.array) -> mx.array:
        return mx.contiguous(t.astype(mx.float32))

    batch, dim, length = u.shape
    dstate = A.shape[1]
    (y,) = _kernel(
        inputs=[
            f32_lmajor(u), f32_lmajor(delta), f32(A), f32_lmajor(B), f32_lmajor(C),
            f32(D), f32_lmajor(z), f32(delta_bias),
        ],
        template=[("N", dstate)],
        grid=(batch * dim, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, length, dim)],
        output_dtypes=[mx.float32],
    )
    return mx.transpose(y, (0, 2, 1))


def install() -> None:
    """Mamba ブロックが参照する selective_scan をこのカーネルに差し替える。"""
    from mlx_speech.models.reuse.mamba import block

    block.selective_scan = selective_scan
