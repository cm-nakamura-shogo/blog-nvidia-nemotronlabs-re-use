#!/usr/bin/env bash
# Brev 上で公式 PyTorch 実装の参照出力と速度を取る
set -euo pipefail
cd "$(dirname "$0")/../.."
source work/.venv-gpu/bin/activate
mkdir -p results

REF=(python scripts/gpu/torch_reference.py --repo-dir models/original)

# 照合用（TF32 なし）
"${REF[@]}" --input work/inputs/rates/*.wav --output-dir work/torch/rates --repeat 6 | tee results/torch_rates.jsonl
"${REF[@]}" --input work/inputs/long/*.wav --output-dir work/torch/long_full --repeat 6 | tee results/torch_long_full.jsonl
"${REF[@]}" --input work/inputs/long/*.wav --output-dir work/torch/long_chunk \
  --chunk-size-in-seconds 5 --hop-length-portion 0.5 --repeat 6 | tee results/torch_long_chunk.jsonl

# 速度の参考（PyTorch 既定の TF32 設定に近い条件）
"${REF[@]}" --input work/inputs/rates/*.wav --output-dir work/torch/rates_tf32 --repeat 6 --allow-tf32 | tee results/torch_rates_tf32.jsonl

# 公式スクリプトそのまま（自作の参照スクリプトと出力が一致するかの確認用。出力は 16bit FLAC）
cd models/original
sh inference.sh
cd ../..
python - <<'EOF'
import numpy as np, soundfile as sf
a, _ = sf.read("models/original/enhanced_audio/mic_test2.flac")
b, _ = sf.read("work/torch/rates/mic_test2_44100.wav")
print({"official_vs_reference_cos": float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)),
       "max_abs": float(np.abs(a - b).max())})
EOF
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
uv pip freeze > results/pip-freeze-gpu.txt
