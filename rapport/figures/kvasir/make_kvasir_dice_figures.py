"""Hardcoded Kvasir-SEG Dice figures for the report.

This script does not read CSV/JSON files. It only uses the summary statistics
reported in the article draft:

- Oracle MedSAM on all Kvasir-SEG images and by split
- GD + MedSAM zero-shot on the test set, threshold 0.10
- GD + MedSAM fine-tuned on the test set, threshold 0.30
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "Kvasir-SEG"
FIGURE_FORMATS = ["pdf", "svg", "png"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ============================================================
# DONNEES HARDCODEES
# ============================================================

ORACLE_GLOBAL = {
    "method": "Oracle MedSAM (GT box)",
    "split": "all",
    "count": 1000,
    "mean": 0.954298,
    "std": 0.024184,
    "min": 0.629385,
    "q1": 0.944214,
    "median": 0.959542,
    "q3": 0.969729,
    "max": 0.992421,
}

ORACLE_BY_SPLIT = {
    "train": {
        "count": 800,
        "mean": 0.954720,
        "std": 0.021593,
        "min": 0.806905,
        "q1": 0.944258,
        "median": 0.959587,
        "q3": 0.969656,
        "max": 0.992421,
    },
    "val": {
        "count": 100,
        "mean": 0.953558,
        "std": 0.022242,
        "min": 0.874240,
        "q1": 0.943697,
        "median": 0.959043,
        "q3": 0.969458,
        "max": 0.981697,
    },
    "test": {
        "count": 100,
        "mean": 0.951666,
        "std": 0.040432,
        "min": 0.629385,
        "q1": 0.941749,
        "median": 0.959893,
        "q3": 0.971275,
        "max": 0.984346,
    },
}

TEST_METHODS = [
    {
        "method": "Oracle MedSAM\n(GT box)",
        "threshold": None,
        "count": 100,
        "mean": ORACLE_BY_SPLIT["test"]["mean"],
        "median": ORACLE_BY_SPLIT["test"]["median"],
        "min": ORACLE_BY_SPLIT["test"]["min"],
        "max": ORACLE_BY_SPLIT["test"]["max"],
    },
    {
        "method": "GD + MedSAM\nzero-shot",
        "threshold": 0.10,
        "count": 100,
        "mean": 0.3231,
        "median": 0.2836,
        "min": 0.0169,
        "max": 0.9805,
    },
    {
        "method": "GD + MedSAM\nfine-tuned",
        "threshold": 0.30,
        "count": 100,
        "mean": 0.9108,
        "median": 0.9445,
        "min": 0.4461,
        "max": 0.9832,
    },
]


# ============================================================
# UTILITAIRES
# ============================================================

def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in FIGURE_FORMATS:
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}", bbox_inches="tight")


def oracle_box_stats(label: str, stats: dict[str, float]) -> dict[str, float | str]:
    return {
        "label": label,
        "whislo": stats["min"],
        "q1": stats["q1"],
        "med": stats["median"],
        "q3": stats["q3"],
        "whishi": stats["max"],
        "mean": stats["mean"],
        "fliers": [],
    }


# ============================================================
# 1. ORACLE MEDSAM - BOXPLOT PAR SPLIT
# ============================================================

split_order = ["train", "val", "test"]
oracle_boxes = [
    oracle_box_stats(f"{split}\n(n={ORACLE_BY_SPLIT[split]['count']})", ORACLE_BY_SPLIT[split])
    for split in split_order
]

fig, ax = plt.subplots(figsize=(6.6, 4.4))

ax.bxp(
    oracle_boxes,
    showmeans=True,
    meanline=True,
    patch_artist=False,
    widths=0.55,
    medianprops={"linewidth": 1.8},
    meanprops={"linestyle": "--", "linewidth": 1.6},
)

ax.set_xlabel("Data split")
ax.set_ylabel("Dice score")
ax.set_title("Oracle MedSAM Dice score by Kvasir-SEG split")
ax.set_ylim(0.60, 1.00)
ax.grid(axis="y", linestyle="--", alpha=0.35)

fig.tight_layout()
save_figure(fig, "fig1")
plt.close(fig)


# ============================================================
# 2. TABLEAU RECAPITULATIF EN FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 2.2))
ax.axis("off")

columns = ["Method", "Threshold", "Mean", "Median", "Min", "Max"]
table_rows = []
for row in TEST_METHODS:
    table_rows.append(
        [
            row["method"].replace("\n", " "),
            "--" if row["threshold"] is None else f"{row['threshold']:.2f}",
            f"{row['mean']:.3f}",
            f"{row['median']:.3f}",
            f"{row['min']:.3f}",
            f"{row['max']:.3f}",
        ]
    )

table = ax.table(
    cellText=table_rows,
    colLabels=columns,
    cellLoc="center",
    loc="center",
    colWidths=[0.34, 0.12, 0.12, 0.12, 0.12, 0.12],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.45)

for (row_idx, _), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.6)
    if row_idx == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#eeeeee")

ax.set_title("Kvasir-SEG test-set Dice summary", pad=12)

fig.tight_layout()
save_figure(fig, "fig2")
plt.close(fig)


# ============================================================
# RESUME CONSOLE
# ============================================================

print("Figures sauvegardees dans :", OUTPUT_DIR)
print("\n=== RESUME TEST SET ===")
for row in TEST_METHODS:
    threshold = "--" if row["threshold"] is None else f"{row['threshold']:.2f}"
    print(
        f"{row['method'].replace(chr(10), ' '):28s} | "
        f"threshold={threshold:>4s} | mean={row['mean']:.4f} | "
        f"median={row['median']:.4f} | min={row['min']:.4f} | max={row['max']:.4f}"
    )
