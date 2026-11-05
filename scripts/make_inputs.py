"""照合と速度計測に使う入力を公式サンプルから作る。

- rates/: 公式サンプル（44.1kHz, 5.1 秒）を各対応レートへリサンプルしたもの
- long/: 同じサンプルを 6 回つないだ約 30 秒の 16kHz 音声（チャンク推論と速度計測用）

    uv run python scripts/make_inputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SRC = Path("models/original/noisy_audio/mic_test2.wav")
OUT = Path("work/inputs")
RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000]


def main() -> None:
    data, sr = sf.read(SRC, dtype="float32")
    (OUT / "rates").mkdir(parents=True, exist_ok=True)
    (OUT / "long").mkdir(parents=True, exist_ok=True)
    for rate in RATES:
        y = data if rate == sr else soxr.resample(data, sr, rate, quality="VHQ").astype(np.float32)
        sf.write(OUT / "rates" / f"mic_test2_{rate}.wav", y, rate, subtype="FLOAT")
    y16 = soxr.resample(data, sr, 16000, quality="VHQ").astype(np.float32)
    sf.write(OUT / "long" / "mic_test2_x6_16000.wav", np.tile(y16, 6), 16000, subtype="FLOAT")


if __name__ == "__main__":
    main()
