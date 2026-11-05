"""MLX 版 SEMamba の部位ごとの所要時間を測る。

    uv run python scripts/profile_mlx.py --sr 44100 --seconds 5.1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

import metal_scan
from mlx_speech.models.reuse import load_mlx_semamba
from mlx_speech.models.reuse.stft import mag_phase_stft, stft_params_for


def timed(label: str, fn, *args, repeat: int = 3):
    out = fn(*args)
    mx.eval(out)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        ts.append(time.perf_counter() - t0)
    print(f"{label:<28} {min(ts) * 1000:9.1f} ms")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--seconds", type=float, default=5.1)
    parser.add_argument("--scan", choices=["metal", "reference"], default="metal")
    args = parser.parse_args()
    if args.scan == "metal":
        metal_scan.install()

    model = load_mlx_semamba(Path("models/mlx"))
    wave = mx.random.normal((1, int(args.sr * args.seconds))) * 0.1
    n_fft, hop, win = stft_params_for(args.sr)
    mag, pha = mag_phase_stft(wave, n_fft, hop, win, compress_factor="relu_log1p")
    print(f"sr={args.sr} n_fft={n_fft} hop={hop} mag={mag.shape}")

    x = mx.concatenate([mx.transpose(mag, (0, 2, 1))[:, None], mx.transpose(pha, (0, 2, 1))[:, None]], axis=1)
    b, c, t, f = x.shape
    x = mx.pad(x, [(0, 0), (0, 0), (0, 2), (0, 2)])
    h = timed("dense_encoder", model.dense_encoder, x)
    print(f"encoded shape (B,C,T,F) = {h.shape}")

    blk = model.TSMamba[0]
    bb, cc, tt, ff = h.shape
    xt = mx.transpose(h, (0, 3, 2, 1)).reshape(bb * ff, tt, cc)
    xf = mx.transpose(h, (0, 2, 3, 1)).reshape(bb * tt, ff, cc)
    timed("time_mamba (1 block)", blk.time_mamba, xt)
    timed("  time fw MambaSSM", blk.time_mamba.forward_blocks, xt)
    timed("freq_mamba (1 block)", blk.freq_mamba, xf)
    timed("  freq fw MambaSSM", blk.freq_mamba.forward_blocks, xf)
    timed("TFMambaBlock (1 block)", blk, h)
    timed("mask_decoder", model.mask_decoder, h)
    timed("phase_decoder", model.phase_decoder, h)
    timed("full model", model, mag, pha, repeat=2)


if __name__ == "__main__":
    main()
