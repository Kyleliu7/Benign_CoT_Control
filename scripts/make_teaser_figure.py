"""Page-1 summary figure: Base vs SFT on ReasonIF and CoT-Control for the three models.

All values are read from results/summary_tables; nothing is typed by hand.

  python scripts/make_teaser_figure.py   ->  reproducible_figures/figure0_teaser.{pdf,png}
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
T = REPO / "results" / "summary_tables"
OUT = REPO / "reproducible_figures"

seeds = pd.read_csv(T / "table26_training_seeds.csv")
olmo = pd.read_csv(T / "table28_olmo_summary.csv")
t1 = pd.read_csv(T / "table1_reasonif_overall.csv")
t2 = pd.read_csv(T / "table2_haskins_standalone.csv")

q_base = seeds[seeds.Condition == "Base"].iloc[0]
q_sft = seeds[seeds.Condition.str.startswith("SFT")]
q_rep = seeds[seeds.Condition == "SFT seed 42"].iloc[0]  # the reported adapter
phi_r = t1[t1.Model == "Phi-4-reasoning"].set_index("Condition")
phi_h = t2[(t2.Model == "Phi-4-reasoning") & (t2.Scoring == "whole trace")].set_index("Condition")
o_base = olmo[olmo.Run == "Base"].iloc[0]
o_sft = olmo[olmo.Run == "SFT lr1e-4"].iloc[0]
N_REASONIF = 300

models = ["Qwen3-14B", "Phi-4-reasoning", "OLMo-3-7B-Think"]
reasonif = {
    "base": [q_base.ReasonIF_IFS_Pct, phi_r.loc["Base (Untouched)", "IFS_Pct"], 100 * o_base.ReasonIF_IFS / N_REASONIF],
    "sft": [q_rep.ReasonIF_IFS_Pct, phi_r.loc["SFT (gpt52-high)", "IFS_Pct"], 100 * o_sft.ReasonIF_IFS / N_REASONIF],
    "seed_points": list(q_sft.ReasonIF_IFS_Pct),
    "fmt": "{:.1f}",
}
haskins = {
    "base": [q_base.Haskins_Mean, phi_h.loc["Base", "Mean_Compliance"], o_base.Haskins_Mean],
    "sft": [q_rep.Haskins_Mean, phi_h.loc["SFT", "Mean_Compliance"], o_sft.Haskins_Mean],
    "seed_points": list(q_sft.Haskins_Mean),
    "olmo_degen0": o_sft.Haskins_Mean_Degenerate_Scored_0,
    "fmt": "{:.2f}",
}

BASE_C, SFT_C = "#9aa5b1", "#2f6db5"
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
w = 0.36
for ax, d, title, ylab in [
    (axes[0], reasonif, "ReasonIF", "Instruction-following success (%)"),
    (axes[1], haskins, "CoT-Control", "Mean compliance"),
]:
    xs = range(len(models))
    ax.bar([x - w / 2 for x in xs], d["base"], w, color=BASE_C, label="Base")
    ax.bar([x + w / 2 for x in xs], d["sft"], w, color=SFT_C, label="SFT on benign reasoning data")
    ax.scatter([w / 2 + w * 0.62] * len(d["seed_points"]), d["seed_points"], s=12, color="black", zorder=3,
               label="Qwen SFT, training seeds 42, 1, 2")
    for x, b, s in zip(xs, d["base"], d["sft"]):
        ax.text(x - w / 2, b + 0.5, d["fmt"].format(b), ha="center", va="bottom", fontsize=7.5)
        ax.text(x + w / 2, s + 0.5, d["fmt"].format(s), ha="center", va="bottom", fontsize=7.5)
    if "olmo_degen0" in d:
        ax.plot([2 + w / 2 - w / 2, 2 + w / 2 + w / 2], [d["olmo_degen0"]] * 2, color="#c0392b", lw=1.6,
                label="OLMo SFT, degenerate traces scored 0")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_ylim(0, max(max(d["sft"]), max(d["seed_points"])) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
handles, labels = [], []
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure0_teaser.{ext}", bbox_inches="tight", dpi=200)
print("Qwen SFT bar is the reported seed-42 adapter (dots: seeds 42, 1, 2); Phi and OLMo are single runs.")
print("ReasonIF", reasonif)
print("Haskins", haskins)
