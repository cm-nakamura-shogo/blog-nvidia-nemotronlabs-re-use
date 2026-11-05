"""NemotronLabs RE-USE を MLX で推論する。

公式 inference.py / inference_chunk.py と同じ前後処理を MLX で再現する。
mlx-speech の REUSEEnhancer.enhance() は 1 秒チャンクの DramaBox 方式で公式と異なるため、
モデルと STFT の部品だけを使い、ループは公式に合わせて書き直している。

    uv run python scripts/mlx_enhance.py --weights models/mlx \
        --input models/original/noisy_audio/mic_test2.wav --output work/mlx/mic_test2.wav
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_speech.models.reuse import load_mlx_semamba
from mlx_speech.models.reuse.stft import (
    mag_phase_istft,
    mag_phase_stft,
    stft_params_for,
    sweep_artifact_filter,
)

COMPRESS_FACTOR = "relu_log1p"


def pad_or_trim(audio: mx.array, length: int, pad_value: float = 1e-8) -> mx.array:
    """公式 utils.pad_or_trim_to_match と同じく、末尾を切るか pad_value で埋める。"""
    if audio.shape[1] >= length:
        return audio[:, :length]
    pad = mx.full((audio.shape[0], length - audio.shape[1]), pad_value, dtype=audio.dtype)
    return mx.concatenate([audio, pad], axis=1)


def enhance_segment(model, wave: mx.array, sr: int, dump: Path | None = None) -> mx.array:
    """(C, T) を 1 回の forward で処理する。公式 inference.py の 1 ファイル分に相当。"""
    n_fft, hop, win = stft_params_for(sr)
    mag, pha = mag_phase_stft(wave, n_fft, hop, win, compress_factor=COMPRESS_FACTOR)
    amp_g, pha_g, _ = model(mag, pha)
    if dump is not None:
        for k, v in {"stft_mag": mag, "stft_pha": pha, "model_amp": amp_g, "model_pha": pha_g}.items():
            np.save(dump / f"{k}.npy", np.array(v))
    amp_g = sweep_artifact_filter(amp_g)
    audio = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, compress_factor=COMPRESS_FACTOR)
    return pad_or_trim(audio, wave.shape[1])


def enhance_chunked(model, wave: mx.array, sr: int, chunk_s: float, hop_portion: float) -> mx.array:
    """公式 inference_chunk.py と同じチャンク分割と Hann 窓の重ね合わせ。"""
    chunk_size = int(chunk_s * sr)
    hop_length = int(hop_portion * chunk_size)
    # torch.hann_window と同じ periodic Hann
    n = mx.arange(chunk_size, dtype=mx.float32)
    window = mx.maximum(0.5 - 0.5 * mx.cos(2 * math.pi * n / chunk_size), 1e-7)

    total = wave.shape[1]
    num_chunks = max(1, math.ceil((total - chunk_size) / hop_length) + 1)
    out = np.zeros((wave.shape[0], total), dtype=np.float32)
    wsum = np.zeros((wave.shape[0], total), dtype=np.float32)
    for i in range(num_chunks):
        if i < num_chunks - 1:
            start = i * hop_length
        else:
            start = max(total - chunk_size, 0)
        seg = wave[:, start : start + chunk_size]
        audio = enhance_segment(model, seg, sr)
        w = window[: audio.shape[1]]
        out[:, start : start + audio.shape[1]] += np.array(audio * w)
        wsum[:, start : start + audio.shape[1]] += np.array(w)
    return mx.array(out / wsum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("models/mlx"))
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size-in-seconds", type=float, default=None)
    parser.add_argument("--hop-length-portion", type=float, default=0.5)
    parser.add_argument("--repeat", type=int, default=1, help="速度計測用。2 回目以降の中央値を記録する")
    parser.add_argument(
        "--scan", choices=["metal", "reference"], default="metal",
        help="reference は mlx-speech の Python ループ実装（長い入力ではメモリが足りない）",
    )
    parser.add_argument("--dump-dir", type=Path, default=None, help="一括処理時の中間出力を npy で保存する")
    parser.add_argument(
        "--cache-limit-gib", type=float, default=1.0,
        help="MLX のバッファキャッシュ上限。無制限だと 44.1kHz 5 秒で 11GiB まで膨らみ、16GB 機ではスワップする",
    )
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_limit_gib * 2**30))
    if args.scan == "metal":
        import metal_scan

        metal_scan.install()

    t0 = time.perf_counter()
    model = load_mlx_semamba(args.weights)
    load_s = time.perf_counter() - t0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in args.input:
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        wave = mx.array(data.T)
        mx.reset_peak_memory()
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            if args.chunk_size_in_seconds is None:
                dump = None
                if args.dump_dir is not None:
                    dump = args.dump_dir / path.stem
                    dump.mkdir(parents=True, exist_ok=True)
                enhanced = enhance_segment(model, wave, sr, dump)
            else:
                enhanced = enhance_chunked(
                    model, wave, sr, args.chunk_size_in_seconds, args.hop_length_portion
                )
            mx.eval(enhanced)
            times.append(time.perf_counter() - t0)
        result = np.array(enhanced).T
        # 照合用に float32 の wav で保存する（公式は 16bit FLAC で保存する）
        sf.write(args.output_dir / f"{path.stem}.wav", result, sr, subtype="FLOAT")
        measured = times[1:] if len(times) > 1 else times
        duration = data.shape[0] / sr
        rec = {
            "file": path.name,
            "sr": sr,
            "channels": data.shape[1],
            "duration_s": round(duration, 3),
            "infer_s_median": round(float(np.median(measured)), 4),
            "rtf": round(float(np.median(measured)) / duration, 4),
            "peak_memory_gib": round(mx.get_peak_memory() / 2**30, 3),
        }
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    print(json.dumps({"load_s": round(load_s, 3)}))


if __name__ == "__main__":
    main()
