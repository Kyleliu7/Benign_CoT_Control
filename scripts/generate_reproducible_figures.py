"""
generate_reproducible_figures.py

Reproducible scientific figure generation pipeline for CoT Controllability study.
Compares Qwen 3 14B and Microsoft Phi-4 Reasoning across:
  1. Haskins 500 crossed transfer matrix (2x2 conditions) with clustered question bootstrap.
  2. ReasonIF 300 benchmark (IFS, Accuracy, Joint Success).
  3. Per-position Forward KL divergence curves (Panel A: cohort mean KL, t=1..100; Panel B: outcome-stratified downstream KL;
     Panels C1/C2: single-trace KL examples).
  4. Appendix task-by-task heatmaps (all 10 tasks x 4 crossed conditions, shared 0-100% scale).

All inputs are read from this repository (see REPO_ROOT below); nothing is hard-coded to a personal machine.
Figure 2 is computed from results/summary_tables/table1_reasonif_overall.csv, which scripts/reproduce_tables.py
regenerates from the raw ReasonIF records. Rows marked Status=UNAVAILABLE (Qwen3-14B base) are omitted, not guessed.
Run `python scripts/reproduce_tables.py --write` first if the summary tables are stale.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm

# Matplotlib publication settings
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['axes.edgecolor'] = '#333333'
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['grid.color'] = '#eaeaea'
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.linewidth'] = 0.6

# Paths (repository-relative)
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_WORKSPACE = str(REPO_ROOT / "reproducible_figures")
OUT_DIR_BRAIN = OUT_DIR_WORKSPACE  # legacy name; there is a single output directory now
BASE_DIR = str(REPO_ROOT)
KL_CACHE_DIR = str(REPO_ROOT / "results" / "kl_divergence" / "caches_float32")
KL_CSV_DIR = str(REPO_ROOT / "results" / "kl_divergence" / "tabular_results_csv")
TABLES_DIR = str(REPO_ROOT / "results" / "summary_tables")

HASKINS_QWEN_DIR = str(REPO_ROOT / "results" / "haskins_500" / "qwen3_14b")
HASKINS_PHI_BUNDLE = str(REPO_ROOT / "results" / "haskins_500" / "phi4_reasoning" / "phi4_haskins_500_bundle.json")

os.makedirs(OUT_DIR_WORKSPACE, exist_ok=True)

# Colorblind-safe palette (Okabe-Ito / Muted)
COND_COLORS = [
    "#4575b4",  # Base -> Base (Cool blue)
    "#74add1",  # Base -> SFT (Light blue)
    "#f46d43",  # SFT -> Base (Coral orange)
    "#d73027"   # SFT -> SFT (Deep red)
]

def save_and_mirror(fig, base_name):
    for out_d in [OUT_DIR_WORKSPACE]:
        for ext in ["png", "pdf", "svg"]:
            p = os.path.join(out_d, f"{base_name}.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved {base_name} (.png, .pdf, .svg)")

def save_csv_and_mirror(df, file_name):
    for out_d in [OUT_DIR_WORKSPACE]:
        df.to_csv(os.path.join(out_d, file_name), index=False)
    print(f"Saved {file_name}")

# ==============================================================================
# 1. HASKINS CROSSED RESULTS (FIGURE 1)
# ==============================================================================
print(">>> Processing Figure 1: Haskins Crossed Results...")

COND_ORDER = ["Base->Base", "Base->SFT", "SFT->Base", "SFT->SFT"]
QWEN_DIRS = {
    "Base->Base": "2x2-base-to-base-on-10tok",
    "Base->SFT": "2x2-base-to-sft-on-10tok",
    "SFT->Base": "2x2-sft-to-base-on-10tok",
    "SFT->SFT": "2x2-sft-to-sft-on-10tok"
}
PHI_KEYS = {
    "Base->Base": "a2",
    "Base->SFT": "a3",
    "SFT->Base": "a1",
    "SFT->SFT": "a4"
}

# Load Qwen individual records
qwen_records_by_prompt = {c: {} for c in COND_ORDER}
all_qwen_subdirs = os.listdir(HASKINS_QWEN_DIR)
for cond, sd_substr in QWEN_DIRS.items():
    matched = [d for d in all_qwen_subdirs if sd_substr in d][0]
    rec_path = os.path.join(HASKINS_QWEN_DIR, matched, "records")
    for f in glob.glob(os.path.join(rec_path, "*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        pidx = d["prompt_idx"]
        if pidx not in qwen_records_by_prompt[cond]:
            qwen_records_by_prompt[cond][pidx] = []
        qwen_records_by_prompt[cond][pidx].append({
            "task": d["task"],
            "comp": float(d.get("continuation_compliance", 0.0)) * 100.0,
            "strict": float(d.get("continuation_score_one", 0.0)) * 100.0
        })

# Load Phi individual records
with open(HASKINS_PHI_BUNDLE, "r", encoding="utf-8") as fp:
    phi_bundle = json.load(fp)

phi_records_by_prompt = {c: {} for c in COND_ORDER}
for it in phi_bundle["items"]:
    pidx = it["prompt_idx"]
    task = it["task"]
    for cond, phi_k in PHI_KEYS.items():
        if pidx not in phi_records_by_prompt[cond]:
            phi_records_by_prompt[cond][pidx] = []
        c_val = float(it["models"][phi_k]["comp"])
        s_val = 100.0 if it["models"][phi_k]["strict"] else 0.0
        phi_records_by_prompt[cond][pidx].append({
            "task": task,
            "comp": c_val,
            "strict": s_val
        })

# Paired cluster bootstrap resampling over the 50 underlying question IDs
N_BOOTSTRAP = 10000
RNG = np.random.default_rng(42)
PROMPT_IDS = np.arange(50)
boot_idx = RNG.choice(PROMPT_IDS, size=(N_BOOTSTRAP, len(PROMPT_IDS)), replace=True)

fig1_rows = []

for model_name, rec_by_prompt in [("Qwen 3 14B", qwen_records_by_prompt), ("Microsoft Phi-4 Reasoning", phi_records_by_prompt)]:
    prompt_means_comp = {c: np.array([np.mean([x["comp"] for x in rec_by_prompt[c][p]]) for p in range(50)]) for c in COND_ORDER}
    prompt_means_strict = {c: np.array([np.mean([x["strict"] for x in rec_by_prompt[c][p]]) for p in range(50)]) for c in COND_ORDER}

    for cond in COND_ORDER:
        comp_all = np.concatenate([np.array([x["comp"] for x in rec_by_prompt[cond][p]]) for p in range(50)])
        strict_all = np.concatenate([np.array([x["strict"] for x in rec_by_prompt[cond][p]]) for p in range(50)])
        
        obs_mean = float(np.mean(comp_all))
        obs_strict_pct = float(np.mean(strict_all))
        obs_strict_count = int(np.round(np.sum(strict_all) / 100.0))

        boot_comp_samples = np.mean(prompt_means_comp[cond][boot_idx], axis=1)
        boot_strict_samples = np.mean(prompt_means_strict[cond][boot_idx], axis=1)

        ci_comp_low, ci_comp_high = np.percentile(boot_comp_samples, [2.5, 97.5])
        ci_strict_low, ci_strict_high = np.percentile(boot_strict_samples, [2.5, 97.5])

        fig1_rows.append({
            "model": model_name,
            "condition": cond,
            "continuation_mean_pct": obs_mean,
            "comp_ci95_low": ci_comp_low,
            "comp_ci95_high": ci_comp_high,
            "strict_count": obs_strict_count,
            "strict_total": 500,
            "strict_pct": obs_strict_pct,
            "strict_ci95_low": ci_strict_low,
            "strict_ci95_high": ci_strict_high
        })

df_fig1 = pd.DataFrame(fig1_rows)
save_csv_and_mirror(df_fig1, "figure1_haskins_crossed_source.csv")

# Plot Figure 1 with clean layout and annotations above error bars
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), dpi=300, facecolor="white")
plt.subplots_adjust(hspace=0.36, wspace=0.24, top=0.91, bottom=0.08)

x_pos = np.arange(len(COND_ORDER))
width = 0.54

# Panel A1: Qwen Continuation Mean
ax = axes[0, 0]
q_data = df_fig1[df_fig1["model"] == "Qwen 3 14B"]
bars = ax.bar(x_pos, q_data["continuation_mean_pct"], width=width, color=COND_COLORS, edgecolor="#222222", linewidth=0.8, zorder=3)
yerr_low = q_data["continuation_mean_pct"] - q_data["comp_ci95_low"]
yerr_high = q_data["comp_ci95_high"] - q_data["continuation_mean_pct"]
ax.errorbar(x_pos, q_data["continuation_mean_pct"], yerr=[yerr_low, yerr_high], fmt="none", ecolor="#111111", elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
ax.set_title("Qwen: Continuation Mean (%)", fontsize=10.5, fontweight="bold", pad=6)
ax.set_ylabel("Compliance (%)", fontsize=9.5)
ax.set_ylim(0, 50)
ax.set_xticks(x_pos)
ax.set_xticklabels(COND_ORDER, fontsize=9)
ax.grid(axis="y", zorder=0)

for bar, (_, row), err_top in zip(bars, q_data.iterrows(), yerr_high):
    h = bar.get_height()
    y_text = h + err_top + 1.2
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, y_text), ha="center", va="bottom", fontsize=8.2, fontweight="bold")

# Panel A2: Qwen Strict Adherence
ax = axes[0, 1]
bars = ax.bar(x_pos, q_data["strict_pct"], width=width, color=COND_COLORS, edgecolor="#222222", linewidth=0.8, zorder=3)
yerr_low = q_data["strict_pct"] - q_data["strict_ci95_low"]
yerr_high = q_data["strict_ci95_high"] - q_data["strict_pct"]
ax.errorbar(x_pos, q_data["strict_pct"], yerr=[yerr_low, yerr_high], fmt="none", ecolor="#111111", elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
ax.set_title("Qwen: Strict Rate (%)", fontsize=10.5, fontweight="bold", pad=6)
ax.set_ylabel("Strict Pass (%)", fontsize=9.5)
ax.set_ylim(0, 32)
ax.set_xticks(x_pos)
ax.set_xticklabels(COND_ORDER, fontsize=9)
ax.grid(axis="y", zorder=0)

for bar, (_, row), err_top in zip(bars, q_data.iterrows(), yerr_high):
    h = bar.get_height()
    y_text = h + err_top + 0.8
    ax.annotate(f"{h:.1f}%\n({int(row['strict_count'])})", xy=(bar.get_x() + bar.get_width()/2, y_text), ha="center", va="bottom", fontsize=7.8, fontweight="bold")

# Panel B1: Phi Continuation Mean
ax = axes[1, 0]
p_data = df_fig1[df_fig1["model"] == "Microsoft Phi-4 Reasoning"]
bars = ax.bar(x_pos, p_data["continuation_mean_pct"], width=width, color=COND_COLORS, edgecolor="#222222", linewidth=0.8, zorder=3)
yerr_low = p_data["continuation_mean_pct"] - p_data["comp_ci95_low"]
yerr_high = p_data["comp_ci95_high"] - p_data["continuation_mean_pct"]
ax.errorbar(x_pos, p_data["continuation_mean_pct"], yerr=[yerr_low, yerr_high], fmt="none", ecolor="#111111", elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
ax.set_title("Phi-4: Continuation Mean (%)", fontsize=10.5, fontweight="bold", pad=6)
ax.set_ylabel("Compliance (%)", fontsize=9.5)
ax.set_ylim(0, 28)
ax.set_xticks(x_pos)
ax.set_xticklabels(COND_ORDER, fontsize=9)
ax.grid(axis="y", zorder=0)

for bar, (_, row), err_top in zip(bars, p_data.iterrows(), yerr_high):
    h = bar.get_height()
    y_text = h + err_top + 0.8
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, y_text), ha="center", va="bottom", fontsize=8.2, fontweight="bold")

# Panel B2: Phi Strict Adherence
ax = axes[1, 1]
bars = ax.bar(x_pos, p_data["strict_pct"], width=width, color=COND_COLORS, edgecolor="#222222", linewidth=0.8, zorder=3)
yerr_low = p_data["strict_pct"] - p_data["strict_ci95_low"]
yerr_high = p_data["strict_ci95_high"] - p_data["strict_pct"]
ax.errorbar(x_pos, p_data["strict_pct"], yerr=[yerr_low, yerr_high], fmt="none", ecolor="#111111", elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
ax.set_title("Phi-4: Strict Rate (%)", fontsize=10.5, fontweight="bold", pad=6)
ax.set_ylabel("Strict Pass (%)", fontsize=9.5)
ax.set_ylim(0, 24)
ax.set_xticks(x_pos)
ax.set_xticklabels(COND_ORDER, fontsize=9)
ax.grid(axis="y", zorder=0)

for bar, (_, row), err_top in zip(bars, p_data.iterrows(), yerr_high):
    h = bar.get_height()
    y_text = h + err_top + 0.6
    ax.annotate(f"{h:.1f}%\n({int(row['strict_count'])})", xy=(bar.get_x() + bar.get_width()/2, y_text), ha="center", va="bottom", fontsize=7.8, fontweight="bold")

fig.suptitle("Haskins 500: 2×2 Crossed Transfer Matrix (95% Cluster-Bootstrap CIs)", fontsize=12, fontweight="bold", y=0.97)

save_and_mirror(fig, "figure1_haskins_crossed")
plt.close()


# ==============================================================================
# 2. REASONIF BENCHMARK (FIGURE 2)
# ==============================================================================
print(">>> Processing Figure 2: ReasonIF Benchmark...")

_t1 = pd.read_csv(os.path.join(TABLES_DIR, "table1_reasonif_overall.csv"))
_t1 = _t1[_t1["Status"].astype(str).str.startswith("OK")]
_COND_MAP = {"Base (Untouched)": "Base", "SFT (gpt52-high)": "SFT",
             "Prefix-OFF (continuation-scored)": "Prefix-OFF", "Prefix-ON (continuation-scored)": "Prefix-ON"}
reasonif_records = [
    {"model": "Qwen 3 14B" if r.Model.startswith("Qwen") else "Microsoft Phi-4 Reasoning",
     "condition": _COND_MAP[r.Condition], "ifs_count": int(r.IFS_Count), "acc_count": int(r.Acc_Count),
     "joint_count": int(r.Joint_Count), "n": int(r.N)}
    for r in _t1.itertuples()
]

def wilson_interval(count, n, conf=0.95):
    p = count / n
    z = norm.ppf(1 - (1 - conf) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return center - margin, center + margin

fig2_table = []
for r in reasonif_records:
    n = r["n"]
    ifs_pct = r["ifs_count"] / n * 100.0
    acc_pct = r["acc_count"] / n * 100.0
    joint_pct = r["joint_count"] / n * 100.0

    ifs_low, ifs_high = wilson_interval(r["ifs_count"], n)
    acc_low, acc_high = wilson_interval(r["acc_count"], n)
    joint_low, joint_high = wilson_interval(r["joint_count"], n)

    fig2_table.append({
        "model": r["model"],
        "condition": r["condition"],
        "n": n,
        "ifs_count": r["ifs_count"],
        "ifs_pct": ifs_pct,
        "ifs_ci95_low": ifs_low * 100.0,
        "ifs_ci95_high": ifs_high * 100.0,
        "acc_count": r["acc_count"],
        "acc_pct": acc_pct,
        "acc_ci95_low": acc_low * 100.0,
        "acc_ci95_high": acc_high * 100.0,
        "joint_count": r["joint_count"],
        "joint_pct": joint_pct,
        "joint_ci95_low": joint_low * 100.0,
        "joint_ci95_high": joint_high * 100.0
    })

df_fig2 = pd.DataFrame(fig2_table)
save_csv_and_mirror(df_fig2, "figure2_reasonif_source.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=300, facecolor="white", sharey=True)
plt.subplots_adjust(wspace=0.15, top=0.80, bottom=0.12)

METRIC_COLORS = ["#0072B2", "#009E73", "#D55E00"]
bar_w = 0.24

# Subplot 1: Qwen 3 14B
q_df = df_fig2[df_fig2["model"] == "Qwen 3 14B"]
q_conds = q_df["condition"].tolist()
x_q = np.arange(len(q_conds))

for i, (m_col, m_low, m_high, label) in enumerate([
    ("ifs_pct", "ifs_ci95_low", "ifs_ci95_high", "IFS (Constraint)"),
    ("acc_pct", "acc_ci95_low", "acc_ci95_high", "Accuracy (Task)"),
    ("joint_pct", "joint_ci95_low", "joint_ci95_high", "Joint Success")
]):
    offset = (i - 1) * bar_w
    vals = q_df[m_col].values
    err_l = vals - q_df[m_low].values
    err_h = q_df[m_high].values - vals
    bars = ax1.bar(x_q + offset, vals, width=bar_w, color=METRIC_COLORS[i], edgecolor="#333333", linewidth=0.8, label=label, zorder=3)
    ax1.errorbar(x_q + offset, vals, yerr=[err_l, err_h], fmt="none", ecolor="#222222", elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
    for bar, val, eh in zip(bars, vals, err_h):
        y_text = val + eh + 2.0
        ax1.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, y_text), ha="center", va="bottom", fontsize=7.5, fontweight="medium")

ax1.set_title("Qwen 3 14B (N = 300)", fontsize=11, fontweight="bold", pad=8)
ax1.set_xticks(x_q)
ax1.set_xticklabels(q_conds, fontsize=9.5)
ax1.set_ylabel("Success Rate (%)", fontsize=10)
ax1.set_ylim(0, 95)
ax1.grid(axis="y", zorder=0)

# Subplot 2: Phi-4 Reasoning
p_df = df_fig2[df_fig2["model"] == "Microsoft Phi-4 Reasoning"]
p_conds = p_df["condition"].tolist()
x_p = np.arange(len(p_conds))

for i, (m_col, m_low, m_high, label) in enumerate([
    ("ifs_pct", "ifs_ci95_low", "ifs_ci95_high", "IFS (Constraint)"),
    ("acc_pct", "acc_ci95_low", "acc_ci95_high", "Accuracy (Task)"),
    ("joint_pct", "joint_ci95_low", "joint_ci95_high", "Joint Success")
]):
    offset = (i - 1) * bar_w
    vals = p_df[m_col].values
    err_l = vals - p_df[m_low].values
    err_h = p_df[m_high].values - vals
    bars = ax2.bar(x_p + offset, vals, width=bar_w, color=METRIC_COLORS[i], edgecolor="#333333", linewidth=0.8, zorder=3)
    ax2.errorbar(x_p + offset, vals, yerr=[err_l, err_h], fmt="none", ecolor="#222222", elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
    for bar, val, eh in zip(bars, vals, err_h):
        y_text = val + eh + 2.0
        ax2.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, y_text), ha="center", va="bottom", fontsize=7.5, fontweight="medium")

ax2.set_title("Phi-4 Reasoning (N = 300)", fontsize=11, fontweight="bold", pad=8)
ax2.set_xticks(x_p)
ax2.set_xticklabels(p_conds, fontsize=9.5)
ax2.grid(axis="y", zorder=0)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
fig.suptitle("ReasonIF Benchmark Performance (Wilson Score 95% CIs)", fontsize=12, fontweight="bold", y=0.98)

save_and_mirror(fig, "figure2_reasonif")
plt.close()


# ==============================================================================
# 3. FORWARD KL DIVERGENCE TRAJECTORIES (FIGURE 3)
# ==============================================================================
print(">>> Processing Figure 3: Per-Position Forward KL Curves...")

def find_kl_path(fname):
    cand = os.path.join(KL_CACHE_DIR, fname)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"Cannot find {fname} in {KL_CACHE_DIR}")

KL_FILES = {
    "Qwen 3 14B - Haskins 500": find_kl_path("kl_cache_qwen3_14b_haskins_500.json"),
    "Qwen 3 14B - ReasonIF 300": find_kl_path("kl_cache_qwen3_14b_reasonif_300.json"),
    "Microsoft Phi-4 - Haskins 500": find_kl_path("kl_cache_phi4_haskins_500.json"),
    "Microsoft Phi-4 - ReasonIF 300": find_kl_path("kl_cache_phi4_reasonif_300.json")
}

CSV_FILES = {
    "Qwen 3 14B - Haskins 500": os.path.join(KL_CSV_DIR, "per_sample_reasoning_kl_qwen3_14b_haskins_500.csv"),
    "Qwen 3 14B - ReasonIF 300": os.path.join(KL_CSV_DIR, "per_sample_reasoning_kl_qwen3_14b_reasonif_300.csv"),
    "Microsoft Phi-4 - Haskins 500": os.path.join(KL_CSV_DIR, "per_sample_reasoning_kl_phi4_haskins_500.csv"),
    "Microsoft Phi-4 - ReasonIF 300": os.path.join(KL_CSV_DIR, "per_sample_reasoning_kl_phi4_reasonif_300.csv")
}

KL_COLORS = {
    "Qwen 3 14B - Haskins 500": "#0072B2",
    "Qwen 3 14B - ReasonIF 300": "#56B4E9",
    "Microsoft Phi-4 - Haskins 500": "#D55E00",
    "Microsoft Phi-4 - ReasonIF 300": "#E69F00"
}

KL_STYLES = {
    "Qwen 3 14B - Haskins 500": "-",
    "Qwen 3 14B - ReasonIF 300": "--",
    "Microsoft Phi-4 - Haskins 500": "-",
    "Microsoft Phi-4 - ReasonIF 300": "--"
}

kl_cohort_data = {}
kl_all512_data = {}
kl_g1_data = {}
kl_g2_data = {}

for label, p in KL_FILES.items():
    with open(p, "r", encoding="utf-8") as fp:
        records = json.load(fp)["records"]
    
    csv_p = CSV_FILES[label]
    df_cohort = pd.read_csv(csv_p) if os.path.exists(csv_p) else pd.DataFrame()
    is_int_key = "reasonif" in label.lower()
    
    cohort_traces = [r["reasoning_kl"] for r in records.values() if len(r.get("reasoning_kl", [])) >= 100]
    cohort_arr = np.array([t[:100] for t in cohort_traces])
    mean_cohort = cohort_arr.mean(axis=0)
    p90_cohort = np.percentile(cohort_arr, 90, axis=0)
    p95_cohort = np.percentile(cohort_arr, 95, axis=0)
    sum_first10 = np.sum(mean_cohort[:10])
    sum_first100 = np.sum(mean_cohort[:100])
    first10_share = (sum_first10 / sum_first100) * 100.0

    kl_cohort_data[label] = {
        "n": len(cohort_traces),
        "mean_curve": mean_cohort,
        "p90_curve": p90_cohort,
        "p95_curve": p95_cohort,
        "first10_share": first10_share,
        "arr": cohort_arr
    }

    # Group 1 (Base FAIL, SFT PASS) and Group 2 (Both PASS)
    if not df_cohort.empty:
        g1_traces = []
        g2_traces = []
        for _, row in df_cohort.iterrows():
            k = str(row["key"]) if is_int_key else row["key"]
            if k in records:
                kl_seq = records[k].get("reasoning_kl", [])
                if len(kl_seq) >= 100:
                    tr = kl_seq[:100]
                    if "Group 1" in str(row.get("quadrant", "")) or row.get("q_code") == "G1":
                        g1_traces.append(tr)
                    elif "Group 2" in str(row.get("quadrant", "")) or row.get("q_code") == "G2":
                        g2_traces.append(tr)
        if g1_traces:
            kl_g1_data[label] = {
                "n": len(g1_traces),
                "mean_curve": np.mean(g1_traces, axis=0),
                "p90_curve": np.percentile(g1_traces, 90, axis=0)
            }
        if g2_traces:
            kl_g2_data[label] = {
                "n": len(g2_traces),
                "mean_curve": np.mean(g2_traces, axis=0)
            }

    all_traces = [r["reasoning_kl"] for r in records.values() if len(r.get("reasoning_kl", [])) > 0]
    max_len = 512
    mean_512 = []
    n_512 = []
    for pos in range(max_len):
        vals_at_pos = [t[pos] for t in all_traces if len(t) > pos]
        if vals_at_pos:
            mean_512.append(np.mean(vals_at_pos))
            n_512.append(len(vals_at_pos))
        else:
            mean_512.append(0.0)
            n_512.append(0)

    kl_all512_data[label] = {
        "mean_512": np.array(mean_512),
        "n_512": np.array(n_512)
    }

fig3_csv_rows = []
for pos in range(1, 101):
    row = {"position": pos}
    for label in KL_FILES:
        row[f"{label}_mean_kl_cohort100"] = kl_cohort_data[label]["mean_curve"][pos - 1]
        row[f"{label}_p90_kl_cohort100"] = kl_cohort_data[label]["p90_curve"][pos - 1]
        if label in kl_g1_data:
            row[f"{label}_g1_mean_kl"] = kl_g1_data[label]["mean_curve"][pos - 1]
    fig3_csv_rows.append(row)

df_fig3 = pd.DataFrame(fig3_csv_rows)
save_csv_and_mirror(df_fig3, "figure3_kl_curves_source.csv")

# Generate Plot for Figure 3:
# Panel A: Cohort Initiation & Early Steering Dynamics (t=1..100, y=0..31 nats)
# Panel B: Downstream Constraint Policing: Focal Signal (Base FAIL, SFT PASS) vs Concordant Pass (t=10..100, y=0..1.05 nats)
# Panel C1: Token-Level Case Study: ReasonIF #84 (punctuation:no_comma, spike to 7.13 nats at t=25)
# Panel C2: Token-Level Case Study: Haskins third_person:4 (third_person, spike to 11.88 nats at t=147)
fig = plt.figure(figsize=(11.5, 10.8), dpi=300, facecolor="white")
gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.15, 1.15], hspace=0.36, wspace=0.22, top=0.93, bottom=0.06, left=0.08, right=0.97)

ax_full = fig.add_subplot(gs[0, :])
ax_policing = fig.add_subplot(gs[1, :])
ax_c1 = fig.add_subplot(gs[2, 0])
ax_c2 = fig.add_subplot(gs[2, 1])

positions_100 = np.arange(1, 101)
zoom_positions = np.arange(10, 101)

# Panel A: Full opening spikes (y: 0 to 31 nats)
phi_h_lbl = "Microsoft Phi-4 - Haskins 500"
qwen_h_lbl = "Qwen 3 14B - Haskins 500"
phi_r_lbl = "Microsoft Phi-4 - ReasonIF 300"
qwen_r_lbl = "Qwen 3 14B - ReasonIF 300"

ax_full.plot(positions_100, kl_cohort_data[qwen_h_lbl]["mean_curve"], color=KL_COLORS[qwen_h_lbl], lw=2.0, label=f"Qwen 3 14B - Haskins 500 (N={kl_cohort_data[qwen_h_lbl]['n']}, First-10 Share={kl_cohort_data[qwen_h_lbl]['first10_share']:.1f}%)")
ax_full.plot(positions_100, kl_cohort_data[qwen_r_lbl]["mean_curve"], color=KL_COLORS[qwen_r_lbl], lw=1.8, linestyle="--", label=f"Qwen 3 14B - ReasonIF 300 (N={kl_cohort_data[qwen_r_lbl]['n']}, First-10 Share={kl_cohort_data[qwen_r_lbl]['first10_share']:.1f}%)")
ax_full.plot(positions_100, kl_cohort_data[phi_h_lbl]["mean_curve"], color=KL_COLORS[phi_h_lbl], lw=2.0, label=f"Microsoft Phi-4 - Haskins 500 (N={kl_cohort_data[phi_h_lbl]['n']}, First-10 Share={kl_cohort_data[phi_h_lbl]['first10_share']:.1f}%)")
ax_full.plot(positions_100, kl_cohort_data[phi_r_lbl]["mean_curve"], color=KL_COLORS[phi_r_lbl], lw=1.8, linestyle="--", label=f"Microsoft Phi-4 - ReasonIF 300 (N={kl_cohort_data[phi_r_lbl]['n']}, First-10 Share={kl_cohort_data[phi_r_lbl]['first10_share']:.1f}%)")

ax_full.axvspan(1, 10, color="#ffffcc", alpha=0.5, zorder=1)
ax_full.axvline(10.5, color="#888888", linestyle=":", lw=1.3, zorder=2)
ax_full.annotate(r"Injected Prefix Window ($t \leq 10$)", xy=(10.5, 20), xytext=(14, 21.5),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                 fontsize=8.8, fontweight="bold", color="#333333")
ax_full.annotate("Qwen: mean KL at t=1 is 25-28 nats", xy=(4, 25), xytext=(23, 26.5),
                 arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.2),
                 fontsize=8.5, fontweight="bold", color="#0072B2",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#e6f2ff", edgecolor="#99ccff", lw=0.8))

ax_full.set_title(r"A: Mean per-position KL (first 100 positions, cohort $L \geq 100$)", fontsize=11, fontweight="bold", pad=6)
ax_full.set_xlabel(r"Reasoning Token Position $t$", fontsize=9.5)
ax_full.set_ylabel("Forward KL [nats]", fontsize=9.5)
ax_full.set_xlim(1, 100)
ax_full.set_ylim(0, 31)
ax_full.grid(True, linestyle=":", alpha=0.6)
ax_full.legend(loc="center right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.2)

# Panel B: Constraint Policing: Base FAIL vs Base PASS
if phi_h_lbl in kl_g1_data:
    ax_policing.plot(zoom_positions, kl_g1_data[phi_h_lbl]["mean_curve"][9:100], linestyle="-", color=KL_COLORS[phi_h_lbl], lw=2.2, label="Phi-4 Haskins G1: Base FAIL, SFT PASS ", zorder=3)
if phi_r_lbl in kl_g1_data:
    ax_policing.plot(zoom_positions, kl_g1_data[phi_r_lbl]["mean_curve"][9:100], linestyle="--", color=KL_COLORS[phi_r_lbl], lw=1.8, label="Phi-4 ReasonIF G1: Base FAIL, SFT PASS ", zorder=3)
if phi_h_lbl in kl_g2_data:
    ax_policing.plot(zoom_positions, kl_g2_data[phi_h_lbl]["mean_curve"][9:100], linestyle=":", color="#888888", lw=1.3, label="Phi-4 Haskins G2: Concordant Pass (Both PASS)", zorder=2)
if qwen_h_lbl in kl_g1_data:
    ax_policing.plot(zoom_positions, kl_g1_data[qwen_h_lbl]["mean_curve"][9:100], linestyle="-", color=KL_COLORS[qwen_h_lbl], lw=1.8, label="Qwen Haskins G1: Base FAIL, SFT PASS", zorder=2)

if phi_h_lbl in kl_g1_data:
    peak_val = kl_g1_data[phi_h_lbl]["mean_curve"][61]
    ax_policing.annotate("Individual-trace spikes are averaged out here", xy=(62, peak_val), xytext=(38, 0.74),
                         arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
                         fontsize=8.5, fontweight="bold", color="#333333",
                         bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffbe6", edgecolor="#dddddd", lw=0.8))

ax_policing.set_title("B: Downstream KL by single-sample outcome group (G1: Base fail / SFT pass; G2: both pass), t = 10-100", fontsize=11, fontweight="bold", pad=6)
ax_policing.set_xlabel(r"Reasoning Token Position $t$", fontsize=9.5)
ax_policing.set_ylabel("Forward KL [nats]", fontsize=9.5)
ax_policing.set_xlim(10, 100)
ax_policing.set_ylim(0, 1.05)
ax_policing.grid(True, linestyle=":", alpha=0.6)
ax_policing.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.0)

# Panel C1: ReasonIF #84 (no_comma)
with open(KL_FILES[phi_r_lbl], "r", encoding="utf-8") as fp:
    r_recs_phi = json.load(fp)["records"]
with open(KL_FILES[qwen_r_lbl], "r", encoding="utf-8") as fp:
    r_recs_qwen = json.load(fp)["records"]

kl_phi_84 = r_recs_phi["84"]["reasoning_kl"]
kl_qwen_84 = r_recs_qwen["84"]["reasoning_kl"]
x84_phi = np.arange(1, len(kl_phi_84) + 1)
x84_qwen = np.arange(1, min(len(kl_qwen_84) + 1, 36))

ax_c1.plot(x84_phi, kl_phi_84, color=KL_COLORS[phi_h_lbl], lw=2.2, label="Phi-4 (SFT=PASS, Base=FAIL)")
ax_c1.plot(x84_qwen, kl_qwen_84[:len(x84_qwen)], color=KL_COLORS[qwen_h_lbl], lw=1.5, linestyle="--", label="Qwen 3 14B (same prompt, own SFT trace)")
ax_c1.annotate("t=25: 7.13 nats", xy=(25, 7.127), xytext=(6, 5.0),
               arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2), fontsize=8.2, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffbe6", edgecolor="#dddddd", lw=0.8))
ax_c1.set_title("C1: KL spike, ReasonIF #84 (punctuation:no_comma)", fontsize=10, fontweight="bold", pad=5)
ax_c1.set_xlabel(r"Reasoning Token Position $t$", fontsize=9)
ax_c1.set_ylabel("Forward KL [nats]", fontsize=9)
ax_c1.set_xlim(1, 35)
ax_c1.set_ylim(0, 8.5)
ax_c1.grid(True, linestyle=":", alpha=0.6)
ax_c1.legend(loc="upper left", fontsize=7.8)

# Panel C2: Haskins third_person:4
with open(KL_FILES[phi_h_lbl], "r", encoding="utf-8") as fp:
    h_recs_phi = json.load(fp)["records"]
with open(KL_FILES[qwen_h_lbl], "r", encoding="utf-8") as fp:
    h_recs_qwen = json.load(fp)["records"]

kl_phi_tp4 = h_recs_phi["third_person:4"]["reasoning_kl"]
kl_qwen_tp4 = h_recs_qwen["third_person:4"]["reasoning_kl"]
xtp4_phi = np.arange(1, len(kl_phi_tp4) + 1)
xtp4_qwen = np.arange(1, min(len(kl_qwen_tp4) + 1, 166))

ax_c2.plot(xtp4_phi, kl_phi_tp4, color=KL_COLORS[phi_h_lbl], lw=2.0, label="Phi-4 (SFT=PASS, Base=FAIL)")
ax_c2.plot(xtp4_qwen, kl_qwen_tp4[:len(xtp4_qwen)], color=KL_COLORS[qwen_h_lbl], lw=1.5, linestyle="--", label="Qwen 3 14B (same prompt, own SFT trace)")
ax_c2.annotate("t=147: 11.88 nats", xy=(147, 11.876), xytext=(45, 9.8),
               arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2), fontsize=8.2, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffbe6", edgecolor="#dddddd", lw=0.8))
ax_c2.set_title("C2: KL spike, Haskins third_person:4", fontsize=10, fontweight="bold", pad=5)
ax_c2.set_xlabel(r"Reasoning Token Position $t$", fontsize=9)
ax_c2.set_ylabel("Forward KL [nats]", fontsize=9)
ax_c2.set_xlim(1, 165)
ax_c2.set_ylim(0, 13.5)
ax_c2.grid(True, linestyle=":", alpha=0.6)
ax_c2.legend(loc="upper left", fontsize=7.8)

fig.suptitle(r"Per-Position Forward KL Trajectories $\mathcal{D}_{\mathrm{KL}}(\pi_{\mathrm{SFT}} \parallel \pi_{\mathrm{Base}})$ & Single-Trace KL Spikes", fontsize=12, fontweight="bold")

save_and_mirror(fig, "figure3_kl_divergence")
plt.close()


# ==============================================================================
# 4. APPENDIX HEATMAPS (FIGURE 4)
# ==============================================================================
print(">>> Processing Figure 4: Appendix Heatmaps...")

ALL_TASKS = [
    "third_person",
    "word_suppression",
    "multiple_word_suppression",
    "arrow_prefix",
    "end_of_sentence",
    "meow_between_words",
    "repeat_sentences",
    "alternating_case",
    "lowercase_thinking",
    "uppercase_thinking"
]

qwen_task_data = {t: {c: {"comp": [], "strict": []} for c in COND_ORDER} for t in ALL_TASKS}
for cond in COND_ORDER:
    for p in range(50):
        for rec in qwen_records_by_prompt[cond][p]:
            t = rec["task"]
            if t in qwen_task_data:
                qwen_task_data[t][cond]["comp"].append(rec["comp"])
                qwen_task_data[t][cond]["strict"].append(rec["strict"])

phi_task_data = {t: {c: {"comp": [], "strict": []} for c in COND_ORDER} for t in ALL_TASKS}
for cond in COND_ORDER:
    for p in range(50):
        for rec in phi_records_by_prompt[cond][p]:
            t = rec["task"]
            if t in phi_task_data:
                phi_task_data[t][cond]["comp"].append(rec["comp"])
                phi_task_data[t][cond]["strict"].append(rec["strict"])

q_mean_matrix = np.zeros((len(ALL_TASKS), len(COND_ORDER)))
q_strict_matrix = np.zeros((len(ALL_TASKS), len(COND_ORDER)))
p_mean_matrix = np.zeros((len(ALL_TASKS), len(COND_ORDER)))
p_strict_matrix = np.zeros((len(ALL_TASKS), len(COND_ORDER)))

fig4_rows = []

for i, t in enumerate(ALL_TASKS):
    for j, c in enumerate(COND_ORDER):
        q_m = np.mean(qwen_task_data[t][c]["comp"])
        q_s = np.mean(qwen_task_data[t][c]["strict"])
        p_m = np.mean(phi_task_data[t][c]["comp"])
        p_s = np.mean(phi_task_data[t][c]["strict"])

        q_mean_matrix[i, j] = q_m
        q_strict_matrix[i, j] = q_s
        p_mean_matrix[i, j] = p_m
        p_strict_matrix[i, j] = p_s

        fig4_rows.append({
            "task": t,
            "condition": c,
            "n": 50,
            "qwen_continuation_mean_pct": q_m,
            "qwen_strict_pct": q_s,
            "phi4_continuation_mean_pct": p_m,
            "phi4_strict_pct": p_s
        })

df_fig4 = pd.DataFrame(fig4_rows)
save_csv_and_mirror(df_fig4, "figure4_appendix_heatmaps_source.csv")

fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2), dpi=300, facecolor="white")
plt.subplots_adjust(hspace=0.28, wspace=0.46, top=0.92, bottom=0.12, left=0.17, right=0.97)

cmap = "YlGnBu"
vmin, vmax = 0.0, 100.0

def plot_heatmap(ax, matrix, title):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, fontsize=9.2, fontweight="medium")
    ax.set_yticks(np.arange(len(ALL_TASKS)))
    ax.set_yticklabels([t.replace("_", " ") for t in ALL_TASKS], fontsize=8.8)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
    
    for r in range(len(ALL_TASKS)):
        for c in range(len(COND_ORDER)):
            val = matrix[r, c]
            txt_color = "white" if val > 55 else "#111111"
            ax.text(c, r, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=8.2, fontweight="medium")
    return im

im1 = plot_heatmap(axes[0, 0], q_mean_matrix, "Qwen: Continuation Mean (%)")
im2 = plot_heatmap(axes[0, 1], q_strict_matrix, "Qwen: Strict Rate (%)")
im3 = plot_heatmap(axes[1, 0], p_mean_matrix, "Phi-4: Continuation Mean (%)")
im4 = plot_heatmap(axes[1, 1], p_strict_matrix, "Phi-4: Strict Rate (%)")

cbar_ax = fig.add_axes([0.25, 0.038, 0.50, 0.018])
cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Adherence Rate (Shared 0–100% Scale, N = 50 per task)", fontsize=9.2, fontweight="medium")
cbar.set_ticks(np.linspace(0, 100, 11))

fig.suptitle("Haskins 500 Task-by-Task 2×2 Transfer Matrices", fontsize=12, fontweight="bold", y=0.97)

save_and_mirror(fig, "figure4_appendix_heatmaps")
plt.close()

print(">>> ALL FIGURES AND SOURCE CSVS GENERATED REPRODUCIBLY.")
