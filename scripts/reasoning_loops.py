"""How often do the s1K teacher-test students loop? For every CoT-Control trace (500 per run) of Base, the s1K Claude 3.7 and
DeepSeek R1 students (Qwen3-14B, Phi-4-reasoning, OLMo-3-7B-Think) and the GPT-5.2 SFT adapter: whether it hit the
generation limit, whether the paper's degenerate-output detector flags it, and a repetition score = share of the 8-grams in
the last 300 words of the trace that already occurred earlier in the same trace (a trace that keeps restating itself
scores near 1). "Looping" = repetition score >= 0.5. Also ReasonIF (300 questions, 16,384-token limit) for the Qwen runs.

  python scripts/reasoning_loops.py      (writes results/summary_tables/reasoning_loops.txt)
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402

THRESH = 0.5


def rep_score(text, n=8, tail=300):
    w = text.split()
    if len(w) < tail + n:
        return 0.0
    head, last = w[:-tail], w[-tail:]
    seen = {tuple(head[i:i + n]) for i in range(len(head) - n + 1)}
    grams = [tuple(last[i:i + n]) for i in range(len(last) - n + 1)]
    return sum(g in seen for g in grams) / len(grams)


def csv_run(pattern):
    f = glob.glob(str(F.RES / pattern))
    assert len(f) == 1, (pattern, f)
    return F._prep(pd.read_csv(f[0]))


HR = F.haskins_runs()
RUNS = {
    ("Qwen3-14B", "Base"): HR["Qwen3-14B Base"], ("Qwen3-14B", "GPT-5.2 SFT"): HR["Qwen3-14B SFT seed 42"],
    ("Qwen3-14B", "s1K Claude 3.7"): csv_run("data_variants/s1k_claude37/haskins_500/qwen3_14b/*/results.csv"),
    ("Qwen3-14B", "s1K R1"): csv_run("data_variants/s1k_deepseek_r1/haskins_500/qwen3_14b/*/results.csv"),
    ("Phi-4-reasoning", "Base"): HR["Phi-4-reasoning Base"], ("Phi-4-reasoning", "GPT-5.2 SFT"): HR["Phi-4-reasoning SFT"],
    ("Phi-4-reasoning", "s1K Claude 3.7"): csv_run("data_variants/phi4_s1k_claude37/haskins_500/phi4_reasoning/*/results.csv"),
    ("Phi-4-reasoning", "s1K R1"): csv_run("data_variants/phi4_s1k_deepseek_r1/haskins_500/phi4_reasoning/*/results.csv"),
    ("OLMo-3-7B-Think", "Base"): HR["OLMo-3-7B-Think Base"], ("OLMo-3-7B-Think", "GPT-5.2 SFT"): HR["OLMo-3-7B-Think SFT lr1e-4"],
    ("OLMo-3-7B-Think", "s1K Claude 3.7"): csv_run("data_variants/olmo3_s1k_claude37/haskins_500/olmo3_7b_think/*/results.csv"),
    ("OLMo-3-7B-Think", "s1K R1"): csv_run("data_variants/olmo3_s1k_deepseek_r1/haskins_500/olmo3_7b_think/*/results.csv"),
}
rows = []
for (model, run), d in RUNS.items():
    rep = d.reasoning.map(rep_score)
    capped = d.capped if "capped" in d else pd.Series(False, index=d.index)
    loop = rep >= THRESH
    rows.append({"Model": model, "Run": run, "N": len(d), "Hit limit": int(capped.sum()),
                 "Looping (rep>=0.5)": int(loop.sum()), "Hit limit and looping": int((capped & loop).sum()),
                 "Degenerate (paper detector)": int(d.degen.sum()), "Median repetition, capped": round(rep[capped].median(), 2) if capped.any() else ""})
pd.set_option("display.width", 250)
out = ["CoT-Control (500 traces per run; generation limit as in each model's evaluation)", pd.DataFrame(rows).to_string(index=False)]

# ReasonIF, Qwen3-14B (reasoning_content of the scored records)
R = F.RES / "reasonif_300"
RIF = {"Base": R / "qwen3_14b/reasonif_qwen3_14b_base_untouched.jsonl", "GPT-5.2 SFT": R / "qwen3_14b/reasonif_qwen3_14b_sft_gpt52_high.jsonl",
       "s1K Claude 3.7": glob.glob(str(F.RES / "data_variants/s1k_claude37/reasonif_300/qwen3_14b/*/scored_responses.jsonl"))[0],
       "s1K R1": glob.glob(str(F.RES / "data_variants/s1k_deepseek_r1/reasonif_300/qwen3_14b/*/scored_responses.jsonl"))[0]}
rows = []
for k, p in RIF.items():
    d = pd.DataFrame([json.loads(x) for x in Path(p).read_text(encoding="utf-8").split("\n") if x.strip()])
    rep = d.reasoning_content.fillna("").astype(str).map(rep_score)
    tr = d.truncated.astype(str).str.lower().eq("true")
    rows.append({"Run": k, "N": len(d), "Truncated": int(tr.sum()), "Looping (rep>=0.5)": int((rep >= THRESH).sum()),
                 "Truncated and looping": int((tr & (rep >= THRESH)).sum()), "Median repetition, truncated": round(rep[tr].median(), 2) if tr.any() else ""})
out += ["", "ReasonIF, Qwen3-14B (300 questions; 16,384-token limit)", pd.DataFrame(rows).to_string(index=False)]
text = "\n".join(out)
print(text)
(F.RES / "summary_tables/reasoning_loops.txt").write_text(text + "\n", encoding="utf-8")

# Does looping drive the teacher result? Seven-task gain with looping traces scored 0, and on pairs where neither trace loops.
from seven_task_composite import SEVEN  # noqa: E402
rows = []
for model in ("Qwen3-14B", "Phi-4-reasoning", "OLMo-3-7B-Think"):
    b, c, r = (RUNS[(model, k)].copy() for k in ("Base", "s1K Claude 3.7", "s1K R1"))
    for d in (b, c, r):
        d["loop"] = d.reasoning.map(rep_score) >= THRESH
        d["primary_l0"] = d.primary.where(~d.loop, 0.0)
    for lab, x, y in (("Claude 3.7 - Base", b, c), ("R1 - Base", b, r), ("Claude 3.7 - R1", r, c)):
        row = {"Model": model, "Contrast": lab}
        for name, col in (("as reported", "primary"), ("looping scored 0", "primary_l0")):
            n, _, _, g, lo, hi = F._paired_gain(x, y, None, col=col, tasks=SEVEN)
            row[name] = f"{g:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        m = x.merge(y, on=["task", "q"], suffixes=("_x", "_y"))
        keep = ~(m.loop_x | m.loop_y) & m.task.isin(SEVEN)
        diff = (m.primary_y - m.primary_x)[keep]
        lo, hi = F._ci(diff.groupby(m.q[keep].values).mean())
        row["pairs where neither loops"] = f"{100 * diff.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] (n={int(keep.sum())})"
        rows.append(row)
extra = ["", "Seven-task contrasts with looping traces handled (looping = repetition score >= 0.5)", pd.DataFrame(rows).to_string(index=False)]
print("\n".join(extra))
with open(F.RES / "summary_tables/reasoning_loops.txt", "a", encoding="utf-8") as fh:
    fh.write("\n".join(extra) + "\n")

# Numbers quoted in the paper (written to results/summary_tables/reasoning_loops.json)
q = RUNS[("Qwen3-14B", "s1K Claude 3.7")]
qrow = next(r for r in rows if r["Model"] == "Qwen3-14B" and r["Contrast"] == "Claude 3.7 - R1")
est, ci = qrow["looping scored 0"].split(" [")
lo, hi = ci.rstrip("]").split(", ")
J = {"qwen_claude37_looping": int((q.reasoning.map(rep_score) >= THRESH).sum()), "qwen_claude37_n": len(q),
     "qwen_claude_minus_r1_loops0": {"est": float(est), "lo": float(lo), "hi": float(hi)}}
(F.RES / "summary_tables/reasoning_loops.json").write_text(json.dumps(J, indent=1), encoding="utf-8")
print(J)
