"""2 つのディレクトリの同名 wav を、時間合わせや正規化なしで比べる。

    uv run python scripts/compare_waveforms.py work/torch/full work/mlx/full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    args = parser.parse_args()

    for ref_path in sorted(args.reference_dir.glob("*.wav")):
        cand_path = args.candidate_dir / ref_path.name
        if not cand_path.exists():
            print(json.dumps({"file": ref_path.name, "error": "missing"}))
            continue
        ref, sr_r = sf.read(ref_path, dtype="float64", always_2d=True)
        cand, sr_c = sf.read(cand_path, dtype="float64", always_2d=True)
        assert sr_r == sr_c and ref.shape == cand.shape, (ref_path, ref.shape, cand.shape)
        r, c = ref.ravel(), cand.ravel()
        diff = c - r
        print(json.dumps({
            "file": ref_path.name,
            "cosine": float(np.dot(r, c) / (np.linalg.norm(r) * np.linalg.norm(c))),
            "relative_rmse": float(np.linalg.norm(diff) / np.linalg.norm(r)),
            "max_abs": float(np.abs(diff).max()),
            "snr_db": float(10 * np.log10(np.sum(r**2) / np.sum(diff**2))),
        }))


if __name__ == "__main__":
    main()
