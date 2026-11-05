"""NemotronLabs RE-USE の重みを MLX 形式に変換する。

mlx-speech の scripts/convert/reuse.py と同じ処理（key 名の付け替えと畳み込み重みの転置）。
変換スクリプトは mlx-speech のパッケージに含まれないため、パッケージ内の関数を呼んで同じことをする。

    uv run python scripts/convert_mlx.py --input-dir models/original --output-dir models/mlx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_speech.models.reuse.loader import assert_keys_match, build_mlx_state
from mlx_speech.models.reuse.semamba import SEMamba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("models/original"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/mlx"))
    args = parser.parse_args()

    state = build_mlx_state(args.input_dir)
    model_keys = {k for k, _ in tree_flatten(SEMamba().parameters())}
    assert_keys_match(model_keys, set(state))

    mx.eval(list(state.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output_dir / "model.safetensors"), state, metadata={"format": "mlx"})
    shutil.copy2(args.input_dir / "config.json", args.output_dir / "config.json")
    print(f"converted {len(state)} keys -> {args.output_dir}")


if __name__ == "__main__":
    main()
