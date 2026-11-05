"""RE-USE（NemotronLabs 版）のモデル構成図を描く。

公式の models/generator_SEMamba_time_d4.py、codec_module_time_d4.py、mamba_block2_SEMamba.py の構造に合わせている。

    uv run python scripts/plot_architecture.py --out results/figures/re-use-architecture.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_IO = "#e8eef7"
C_ENC = "#dcefe3"
C_MAMBA = "#fde9c8"
C_DEC = "#f3dde6"
C_EDGE = "#4a4a4a"


def box(ax, x, y, w, h, text, color, fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=C_EDGE, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.2, color=C_EDGE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # 上段：全体の流れ
    y, h = 4.2, 1.6
    box(ax, 0.05, y, 2.0, h, "劣化音声\n↓\nSTFT\n（窓長40ms、ホップ5ms）", C_IO, 9)
    box(ax, 2.4, y, 1.6, h, "入力\n振幅（log1p圧縮）\n位相\n2ch × 時間 × 周波数", C_IO, 9)
    box(ax, 4.35, y, 1.95, h, "エンコーダ\n畳み込み＋Dense Block\n時間 1/4、周波数 1/2\n64ch", C_ENC, 9)
    box(ax, 6.65, y, 1.95, h, "TFMambaブロック\n× 30段", C_MAMBA, 11)
    box(ax, 9.0, y + 0.85, 2.0, 0.75, "振幅デコーダ\n→ 振幅", C_DEC, 9)
    box(ax, 9.0, y, 2.0, 0.75, "位相デコーダ\n→ 実部・虚部 → atan2", C_DEC, 9)
    box(ax, 11.4, y, 1.5, h, "ISTFT\n↓\n強調後の音声", C_IO, 9)
    arrow(ax, 2.05, y + h / 2, 2.4, y + h / 2)
    arrow(ax, 4.0, y + h / 2, 4.35, y + h / 2)
    arrow(ax, 6.3, y + h / 2, 6.65, y + h / 2)
    arrow(ax, 8.6, y + h / 2, 9.0, y + 1.22)
    arrow(ax, 8.6, y + h / 2, 9.0, y + 0.38)
    arrow(ax, 11.0, y + 1.22, 11.4, y + h / 2)
    arrow(ax, 11.0, y + 0.38, 11.4, y + h / 2)
    ax.text(10.0, y - 0.25, "デコーダは時間 ×4、周波数 ×2 で元のサイズに戻す", ha="center", fontsize=8.5)

    # 下段：TFMambaブロックの中身
    ax.add_patch(FancyBboxPatch((0.6, 0.25), 11.8, 3.05, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#fffaf0", ec=C_EDGE, lw=1.0, ls="--"))
    ax.text(0.85, 2.95, "TFMambaブロック（1段分）", fontsize=10.5, va="center")
    arrow(ax, 7.6, y, 6.5, 3.3)

    by, bh = 0.75, 1.7
    box(ax, 0.85, by, 1.95, bh, "入力\n64ch\n× 時間 × 周波数", "#ffffff", 9)
    box(ax, 3.1, by, 3.6, bh, "時間方向の双方向Mamba\n周波数ビンごとに時間軸に沿って\n順方向Mamba ＋ 逆方向Mamba\n→ 結合 → 線形層 → LayerNorm", C_MAMBA, 9)
    box(ax, 7.2, by, 3.6, bh, "周波数方向の双方向Mamba\nフレームごとに周波数軸に沿って\n順方向Mamba ＋ 逆方向Mamba\n→ 結合 → 線形層 → LayerNorm", C_MAMBA, 9)
    box(ax, 11.1, by, 1.1, bh, "出力", "#ffffff", 9)
    arrow(ax, 2.8, by + bh / 2, 3.1, by + bh / 2)
    arrow(ax, 6.7, by + bh / 2, 7.2, by + bh / 2)
    arrow(ax, 10.8, by + bh / 2, 11.1, by + bh / 2)
    ax.text(4.9, by - 0.28, "残差接続あり", ha="center", fontsize=8.5)
    ax.text(9.0, by - 0.28, "残差接続あり", ha="center", fontsize=8.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
