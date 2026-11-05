"""Metal カーネル版 scan が mlx-speech の参照実装と一致するかを乱数入力で確かめる。

    uv run python scripts/test_metal_scan.py
"""

from __future__ import annotations

import mlx.core as mx

from mlx_speech.models.reuse.mamba.scan import selective_scan as reference
from metal_scan import selective_scan as metal


def main() -> None:
    mx.random.seed(0)
    for batch, dim, length, dstate in [(1, 8, 5, 16), (3, 256, 200, 16), (7, 256, 513, 16)]:
        u = mx.random.normal((batch, dim, length))
        delta = mx.random.normal((batch, dim, length)) * 3  # softplus の両端も通す
        A = -mx.exp(mx.random.normal((dim, dstate)))
        B = mx.random.normal((batch, dstate, length))
        C = mx.random.normal((batch, dstate, length))
        D = mx.random.normal((dim,))
        z = mx.random.normal((batch, dim, length))
        bias = mx.random.normal((dim,))
        ref = reference(u, delta, A, B, C, D=D, z=z, delta_bias=bias)
        got = metal(u, delta, A, B, C, D=D, z=z, delta_bias=bias)
        max_abs = mx.max(mx.abs(ref - got)).item()
        rel = (mx.linalg.norm(ref - got) / mx.linalg.norm(ref)).item()
        print(f"shape=({batch},{dim},{length},{dstate}) max_abs={max_abs:.3e} rel_l2={rel:.3e}")
        assert rel < 1e-5, rel


if __name__ == "__main__":
    main()
