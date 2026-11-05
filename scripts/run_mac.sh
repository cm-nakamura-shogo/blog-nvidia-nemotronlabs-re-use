#!/usr/bin/env bash
# NemotronLabs RE-USE の重みを取得して MLX に変換し、公式サンプル音声で推論と速度計測を行う
set -euo pipefail
cd "$(dirname "$0")/.."

# 重みとサンプル音声を revision 固定で取得する
uv run python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/NVIDIA-NemotronLabs-RE-USE",
                  revision="31f82f2efad1095034492f236e8dac9239ba713b",
                  allow_patterns=["model.safetensors", "config.json", "noisy_audio/*"],
                  local_dir="models/original")
EOF

uv run python scripts/convert_mlx.py --input-dir models/original --output-dir models/mlx
shasum -a 256 models/original/model.safetensors models/mlx/model.safetensors

# 1 ファイル一括（公式 inference.sh 相当）
uv run python scripts/mlx_enhance.py --weights models/mlx \
  --input models/original/noisy_audio/mic_test2.wav \
  --output-dir work/mlx/full --repeat 6

# チャンク推論（公式 inference_chunk.sh 相当: 5 秒, hop 0.5）
uv run python scripts/mlx_enhance.py --weights models/mlx \
  --input models/original/noisy_audio/mic_test2.wav \
  --output-dir work/mlx/chunk --chunk-size-in-seconds 5 --hop-length-portion 0.5 --repeat 6
