# NVIDIA NemotronLabs RE-USE の検証

NVIDIAの音声強調モデル [nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE)（9.6Mパラメータ、OpenMDW-1.1）を、次の2つの環境で動かした検証コードと測定結果です。

- NVIDIA L40S（Brev）で、公式のPyTorch実装（mamba_ssm）を動かす
- Apple Silicon（M4、16GB）で、重みをMLXに変換して動かす

> [!NOTE]
> このリポジトリは、検証ブログ記事のコードと検証記録です。
> モデルの重み、音声ファイル、記事の下書きは含めていません。
> 重みと公式のサンプル音声は、各スクリプトがHugging Faceからrevisionを固定して取得します。

モデルのrevisionは `31f82f2efad1095034492f236e8dac9239ba713b` に固定しています。

## 結果の概要

公式のサンプル音声（44.1kHz、5.1秒）を各サンプリングレートにリサンプルして測りました。
RTFは処理時間を音声の長さで割った値で、初回を除いた中央値です。

| サンプリングレート | L40S PyTorch RTF | L40S GPUメモリ | M4 MLX RTF | 公式とMLXの一致度（SNR） |
| --- | ---: | ---: | ---: | ---: |
| 8kHz | 0.016 | 0.49GiB | 0.24 | 41.6dB |
| 16kHz | 0.030 | 0.92GiB | 0.46 | 41.0dB |
| 22.05kHz | 0.045 | 1.25GiB | 0.62 | 41.5dB |
| 24kHz | 0.045 | 1.35GiB | 0.67 | 48.1dB |
| 32kHz | 0.061 | 1.78GiB | 0.89 | 46.9dB |
| 44.1kHz | 0.092 | 2.45GiB | 1.26 | 65.6dB |
| 48kHz | 0.091 | 2.65GiB | 1.37 | 35.1dB |

- L40SはFP32（TF32無効）、GPUメモリは `torch.cuda.max_memory_allocated` の値です。
- MLX版と公式の出力は、全レートでコサイン類似度0.99985以上でした。
- サンプリングレートによって一致度がばらつくのは、振幅がほぼゼロの周波数ビンで位相が丸め誤差によってずれ、モデルがその位相を入力に使うためです。同じMLX実装でも、入力に1e-8の白色雑音を足すだけで出力はSNR 33〜37dBまで変わるので、この差はモデル自体の入力感度の範囲に収まっています。

生データは `results/` にあります。

| ファイル | 内容 |
| --- | --- |
| `results/torch_*.jsonl` | L40Sの速度とメモリ |
| `results/mlx_*.jsonl` | M4の速度とメモリ |
| `results/parity.jsonl` | 波形の照合結果（公式とMLX、TF32の有無、入力に微小雑音を足した場合） |
| `results/pip-freeze-gpu.txt` | L40S側のPython環境 |
| `results/figures/` | スペクトログラムとモデル構成の図 |

## ディレクトリ構成

```text
scripts/
├── run_mac.sh              # Mac: 重みの取得、MLXへの変換、サンプルでの推論
├── convert_mlx.py          # Mac: 重みをMLX形式に変換する
├── mlx_enhance.py          # Mac: 公式と同じ前後処理でMLX推論する（一括、チャンク）
├── metal_scan.py           # Mac: selective scan の Metal カーネル（mlx-speech の実装を差し替える）
├── test_metal_scan.py      # Mac: Metal カーネルと mlx-speech の参照実装の照合
├── profile_mlx.py          # Mac: 部位ごとの所要時間の計測
├── make_inputs.py          # 照合と速度計測用の入力（各サンプリングレート、30秒）を作る
├── compare_waveforms.py    # 2つのディレクトリの同名wavを比べる
├── plot_spectrogram.py     # スペクトログラムの図
├── plot_architecture.py    # モデル構成の図
├── gpu/
│   ├── setup_brev.sh       # L40S: 公式実装の実行環境を作る
│   ├── torch_reference.py  # L40S: 公式と同じ処理で推論し、float32で保存して速度を測る
│   └── run_reference.sh    # L40S: 照合用の出力と速度の一式を取る
└── host/
    ├── sync_to_brev.sh     # 手元からBrevへスクリプトと入力を送る
    └── remote.sh           # Brevでのコマンド実行と結果の取得
```

## Macでの実行（MLX）

Apple SiliconのMacと [uv](https://docs.astral.sh/uv/) が必要です。
MLXの実装には [mlx-speech](https://github.com/appautomaton/mlx-speech) を使い、commit `503bd7e` に固定しています。

```bash
uv sync
bash scripts/run_mac.sh
```

`run_mac.sh` は、重みと公式サンプルの取得、MLXへの変換、サンプル音声での推論と速度計測を順に行います。
出力は `work/mlx/` に保存されます。

mlx-speechの selective scan はPythonのループで実装されていて、長い入力では中間テンソルが大きくなりすぎて動きません。
`scripts/mlx_enhance.py` は、実行時にこれを `scripts/metal_scan.py` の Metal カーネルに差し替えます（`--scan reference` で元の実装を使えます）。
また、MLXのバッファキャッシュが膨らんで16GB機でスワップするため、上限を既定で1GiBにしています（`--cache-limit-gib`）。

Metal カーネルの照合と、各サンプリングレートでの計測は以下のとおりです。

```bash
uv run python scripts/test_metal_scan.py
uv run python scripts/make_inputs.py
uv run python scripts/mlx_enhance.py --input work/inputs/rates/*.wav --output-dir work/mlx/rates --repeat 3
```

## L40Sでの実行（公式PyTorch実装）

[NVIDIA Brev](https://developer.nvidia.com/brev) のL40Sインスタンス（`l40s-48gb.1x`）で動かしました。
環境は、NVIDIAドライバ565.57.01、Python 3.10（uv管理）、PyTorch 2.6.0+cu124、mamba_ssm 2.2.4、causal-conv1d 1.5.0.post8です。

```bash
brev create cm-nakamura-re-use --type l40s-48gb.1x

# 手元で入力を作ってから送る
uv run python scripts/make_inputs.py
bash scripts/host/sync_to_brev.sh

# Brev側で環境を作り、測定する
bash scripts/host/remote.sh run "bash scripts/gpu/setup_brev.sh"
bash scripts/host/remote.sh run "bash scripts/gpu/run_reference.sh"

# 結果を取得して、MLX版と照合する
bash scripts/host/remote.sh fetch work/torch results
uv run python scripts/compare_waveforms.py work/torch/rates work/mlx/rates
```

workspace名は環境変数 `REUSE_BREV_WORKSPACE` で変えられます。
`remote.sh run` は、仮想環境（`work/.venv-gpu`）があればそれを有効にしてからコマンドを実行します。

環境構築では、以下の点に注意が必要でした。

- torchと他のパッケージを分けてインストールすると、torchが新しい版（cu130）に上がってドライバ565で動かなくなるため、1回の `uv pip install` で解決させる
- mamba_ssm 2.3系はtriton 3.5以上を要求してtorch 2.6と両立しないため、2.2.4を使う
- tritonが実行時にC拡張をビルドするため、ヘッダ（`Python.h`）を同梱したuv管理のPythonを使う

## ライセンス

このリポジトリのコードはMITライセンスです（[LICENSE](LICENSE)）。
NVIDIAの推論コードの処理を再現した部分などの出典は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に記載しています。

モデルの重みはOpenMDW-1.1で配布されており、このリポジトリには含めていません。
