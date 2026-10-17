"""Figures 1 and 3 of the revision with CoT-Control on the seven non-character tasks (the paper's main CoT-Control number).

  figure0_teaser7          Base vs SFT on ReasonIF and seven-task CoT-Control for the three models (same layout as
                           make_teaser_figure.py; ReasonIF values from the same summary tables)
  figure1_cotcontrol_crossed7  crossed donor-recipient continuation mean and strict rate on the seven tasks (350 items per
                           condition), Qwen3-14B and Phi-4-reasoning, 95% cluster-bootstrap intervals over the 50 questions
                           (same layout and colors as generate_reproducible_figures.py)

All values are computed from the raw records; nothing is typed by hand.

  python scripts/make_seven_task_figures.py   ->  reproducible_figures/<name>.{pdf,png} and <name>_source.csv
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
import analysis_prefix_length_paired as A  # noqa: E402
from haskins_header_stripped import runs  # noqa: E402
from seven_task_composite import SEVEN  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
T = REPO / "results" / "summary_tables"
OUT = REPO / "reproducible_figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Figure 1 (teaser)
H = runs()
H.update({k: v for k, v in F.haskins_runs().items() if not k.startswith("Qwen3-14B")})
for d in H.values():
    d["primary_d0"] = np.where(d.degen, 0.0, d.primary)


def m7(k, col="primary"):
    d = H[k]
    return 100 * d[d.task.isin(SEVEN)][col].mean()


seeds = pd.read_csv(T / "table26_training_seeds.csv")
olmo = pd.read_csv(T / "table28_olmo_summary.csv")
t1 = pd.read_csv(T / "table1_reasonif_overall.csv")
q_base = seeds[seeds.Condition == "Base"].iloc[0]
q_sft = seeds[seeds.Condition.str.startswith("SFT")]
q_rep = seeds[seeds.Condition == "SFT seed 42"].iloc[0]
phi_r = t1[t1.Model == "Phi-4-reasoning"].set_index("Condition")
o_base = olmo[olmo.Run == "Base"].iloc[0]
o_sft = olmo[olmo.Run == "SFT lr1e-4"].iloc[0]
N_REASONIF = 300
models = ["Qwen3-14B", "Phi-4-reasoning", "OLMo-3-7B-Think"]
reasonif = {"base": [q_base.ReasonIF_IFS_Pct, phi_r.loc["Base (Untouched)", "IFS_Pct"], 100 * o_base.ReasonIF_IFS / N_REASONIF],
            "sft": [q_rep.ReasonIF_IFS_Pct, phi_r.loc["SFT (gpt52-high)", "IFS_Pct"], 100 * o_sft.ReasonIF_IFS / N_REASONIF],
            "seed_points": list(q_sft.ReasonIF_IFS_Pct), "fmt": "{:.1f}"}
cot = {"base": [m7("Base"), m7("Phi-4-reasoning Base"), m7("OLMo-3-7B-Think Base")],
       "sft": [m7("SFT seed 42"), m7("Phi-4-reasoning SFT"), m7("OLMo-3-7B-Think SFT lr1e-4")],
       "seed_points": [m7("SFT seed 42"), m7("SFT seed 1"), m7("SFT seed 2")],
       "olmo_degen0": m7("OLMo-3-7B-Think SFT lr1e-4", "primary_d0"), "fmt": "{:.2f}"}
pd.DataFrame({"model": models, "reasonif_base": reasonif["base"], "reasonif_sft": reasonif["sft"], "cot7_base": cot["base"],
              "cot7_sft": cot["sft"]}).to_csv(OUT / "figure0_teaser7_source.csv", index=False)

BASE_C, SFT_C = "#9aa5b1", "#2f6db5"
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
w = 0.36
for ax, d, title, ylab in [(axes[0], reasonif, "ReasonIF", "Instruction-following success (%)"),
                           (axes[1], cot, "CoT-Control (seven non-character tasks)", "Mean compliance")]:
    xs = range(len(models))
    ax.bar([x - w / 2 for x in xs], d["base"], w, color=BASE_C, label="Base")
    ax.bar([x + w / 2 for x in xs], d["sft"], w, color=SFT_C, label="SFT on benign reasoning data")
    ax.scatter([w / 2 + w * 0.62] * len(d["seed_points"]), d["seed_points"], s=12, color="black", zorder=3,
               label="Qwen SFT, training seeds 42, 1, 2")
    for x, b, s in zip(xs, d["base"], d["sft"]):
        ax.text(x - w / 2, b + 0.5, d["fmt"].format(b), ha="center", va="bottom", fontsize=7.5)
        ax.text(x + w / 2, s + 0.5, d["fmt"].format(s), ha="center", va="bottom", fontsize=7.5)
    if "olmo_degen0" in d:
        ax.plot([2, 2 + w], [d["olmo_degen0"]] * 2, color="#c0392b", lw=1.6, label="OLMo SFT, degenerate traces scored 0")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_ylim(0, max(max(d["sft"]), max(d["seed_points"])) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
handles, labels = [], []
for ax in axes:
    for h, lab in zip(*ax.get_legend_handles_labels()):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure0_teaser7.{ext}", bbox_inches="tight", dpi=200)
plt.close(fig)
print("teaser ReasonIF", reasonif["base"], reasonif["sft"], "| CoT-Control 7", [round(x, 2) for x in cot["base"]], [round(x, 2) for x in cot["sft"]])

# ---------------------------------------------------------------- Figure 3 (crossed, seven tasks)
COND = [("BB", "Base->Base"), ("BS", "Base->SFT"), ("SB", "SFT->Base"), ("SS", "SFT->SFT")]
COLORS = ["#4575b4", "#74add1", "#f46d43", "#d73027"]
L = A.load_long()
L = L[L.task.isin(SEVEN)]
rows = []
boot = np.random.default_rng(42).integers(0, 50, size=(10_000, 50))
for model, g in L.groupby("model"):
    for key, lab in COND:
        c = g[g.cond == key]
        pq = c.groupby("q")[["comp", "strict"]].mean().sort_index()
        assert len(pq) == 50
        mb = pq.comp.to_numpy()[boot].mean(1) * 100
        sb = pq.strict.to_numpy()[boot].mean(1) * 100
        rows.append({"model": model, "condition": lab, "n": len(c), "mean": 100 * c.comp.mean(), "mean_lo": np.percentile(mb, 2.5),
                     "mean_hi": np.percentile(mb, 97.5), "strict_count": int(c.strict.sum()), "strict_pct": 100 * c.strict.mean(),
                     "strict_lo": np.percentile(sb, 2.5), "strict_hi": np.percentile(sb, 97.5)})
df = pd.DataFrame(rows)
df.to_csv(OUT / "figure1_cotcontrol_crossed7_source.csv", index=False)
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), dpi=300, facecolor="white")
plt.subplots_adjust(hspace=0.36, wspace=0.24, top=0.91, bottom=0.08)
x = np.arange(4)
for r, (model, name) in enumerate([("Qwen3-14B", "Qwen3-14B"), ("Phi-4-reasoning", "Phi-4-reasoning")]):
    q = df[df.model == model]
    for cidx, (col, lo, hi, title, ylab, strict) in enumerate([("mean", "mean_lo", "mean_hi", "Continuation Mean (%)", "Compliance (%)", False),
                                                                 ("strict_pct", "strict_lo", "strict_hi", "Strict Rate (%)", "Strict Pass (%)", True)]):
        ax = axes[r, cidx]
        bars = ax.bar(x, q[col], width=0.54, color=COLORS, edgecolor="#222222", linewidth=0.8, zorder=3)
        el, eh = q[col] - q[lo], q[hi] - q[col]
        ax.errorbar(x, q[col], yerr=[el, eh], fmt="none", ecolor="#111111", elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
        ax.set_title(f"{name}: {title}", fontsize=10.5, fontweight="bold", pad=6)
        ax.set_ylabel(ylab, fontsize=9.5)
        ax.set_ylim(0, float((q[hi].max()) * 1.35))
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in COND], fontsize=9)
        ax.grid(axis="y", color="#eaeaea", linestyle="--", linewidth=0.6, zorder=0)
        for bar, (_, row), top in zip(bars, q.iterrows(), eh):
            h = bar.get_height()
            txt = f"{h:.1f}%\n({int(row['strict_count'])})" if strict else f"{h:.1f}%"
            ax.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, h + top + 0.6), ha="center", va="bottom", fontsize=8, fontweight="bold")
fig.suptitle("CoT-Control, seven non-character tasks: Crossed Donor→Recipient Runs (95% Cluster-Bootstrap CIs)", fontsize=12, fontweight="bold", y=0.97)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure1_cotcontrol_crossed7.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(df.round(2).to_string(index=False))
