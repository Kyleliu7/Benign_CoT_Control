"""Numbers requested by the October 2026 review, all recomputed from raw records in the repository (CPU only).

  A  Degenerate-trace cutoff sensitivity: CoT-Control gain over Base with degenerate traces scored 0, for word cutoffs
     20 (the paper's rule), 30 and 50 (the meow rule unchanged).
  B  Standalone CoT-Control per task (mean and strict) for Qwen3-14B Base / SFT seeds and Phi-4-reasoning Base / SFT.
  C  ReasonIF Prefix-OFF / Prefix-ON (Qwen and Phi): forced prefixes shorter than 10 tokens and why; IFS as stored,
     on prefix + continuation ("official") and on the continuation only; paired differences with McNemar p.
  D  Crossed CoT-Control runs (Qwen): prefixes that already contain </think>, unfinished continuations, and the
     recipient effect (SFT->SFT minus SFT->Base) on items where neither applies.
  E  Fixed-opening controls (Qwen, eight style tasks): continuation length per opening and the gain at an equal
     100-word window.
  F  ReasonIF all-capitals items: share of letters in the reasoning that are uppercase (why 0 passes).
  G  Across runs and tasks: does the rate of openings that name the constraint predict the CoT-Control gain?
  H  Training data: how often the opening heading and first sentence restate the user's request.
  I  Existing KL caches: KL at the first reasoning positions for SFT traces that open with a bold heading vs not.

  python scripts/review_followups.py > results/summary_tables/review_followups.txt
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis_followups as F  # noqa: E402
import analysis_prefix_length_paired as P  # noqa: E402
from evaluators.haskins_evaluator import grade_paper_protocol, primary_score  # noqa: E402
from haskins_header_stripped import rescore, runs as h_runs  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)
RES = F.RES
LEAD = re.compile(r"^\s*\*\*[^*\n]{1,120}\*\*\s*")


def ci(v):
    return F._ci(np.asarray(v, float))


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    b01, b10 = int((~a & b).sum()), int((a & ~b).sum())
    return b01, b10, F._mcnemar(b01, b10)


def section(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


# ------------------------------------------------------------------ A
section("A. Degenerate cutoff sensitivity (CoT-Control mean, degenerate traces scored 0; gain vs the model's Base)")
H = F.haskins_runs()
pairs = [("Qwen3-14B Base", k) for k in H if k.startswith("Qwen3-14B SFT")] + \
        [("Phi-4-reasoning Base", "Phi-4-reasoning SFT")] + [("OLMo-3-7B-Think Base", k) for k in H if k.startswith("OLMo-3-7B-Think SFT")]


def degen(task, text, cut):
    w = str(text).split()
    return len(w) < cut or (task == "meow_between_words" and len(w) > 0 and sum(x.strip(".,!").lower() == "meow" for x in w) / len(w) > 0.9)


rows = []
for b, s in pairs:
    m = H[b].merge(H[s], on=["task", "q"], suffixes=("_b", "_s"))
    row = {"Model": s, "Gain (no filter)": round(100 * (m.primary_s - m.primary_b).mean(), 2)}
    for cut in (20, 30, 50):
        db = np.array([degen(t, x, cut) for t, x in zip(m.task, m.reasoning_b)])
        ds = np.array([degen(t, x, cut) for t, x in zip(m.task, m.reasoning_s)])
        d = np.where(ds, 0, m.primary_s) - np.where(db, 0, m.primary_b)
        lo, hi = ci(pd.Series(d).groupby(m.q.values).mean())
        row[f"<{cut} words: degenerate B/S"] = f"{db.sum()}/{ds.sum()}"
        row[f"<{cut} words: gain [95% CI]"] = f"{100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ B
section("B. Standalone CoT-Control per task: mean % / strict passes (of 50)")
cols = {k: H[k] for k in H if k.startswith("Qwen3-14B") or k.startswith("Phi-4")}
tab = pd.DataFrame({k: v.groupby("task").apply(lambda g: f"{100 * g.primary.mean():.1f} / {int(g.strict.sum())}") for k, v in cols.items()})
print(tab.to_string())

# ------------------------------------------------------------------ C
section("C. ReasonIF prefix runs: short prefixes and scoring basis (the paper reports continuation-only scores)")


def tb(v):
    return str(v).strip().lower() == "true"


R = F.reasonif_runs()
pref = {"Qwen3-14B Prefix-OFF": F._jl(RES / "reasonif_300/qwen3_14b/reasonif_qwen3_14b_prefix_constraint_off.jsonl"),
        "Qwen3-14B Prefix-ON": F._jl(RES / "reasonif_300/qwen3_14b/reasonif_qwen3_14b_prefix_constraint_on.jsonl"),
        "Phi-4-reasoning Prefix-OFF": F._jl(RES / "reasonif_300/phi4_reasoning/prefix_off/scored_responses.jsonl"),
        "Phi-4-reasoning Prefix-ON": F._jl(RES / "reasonif_300/phi4_reasoning/prefix_on/scored_responses.jsonl")}
rows = []
for name, recs in pref.items():
    model = name.split(" Prefix")[0]
    base = {int(r["dataset_index"]): tb(r["instruction_following"]) for r in R[f"{model} Base"][1]}
    sft = {int(r["dataset_index"]): tb(r["instruction_following"]) for r in R[f"{model} SFT" if model.startswith("Phi") else f"{model} SFT seed 42"][1]}
    short = [r for r in recs if len(json.loads(r["forced_prefix_token_ids"]) if isinstance(r["forced_prefix_token_ids"], str) else r["forced_prefix_token_ids"]) < 10]
    why = pd.Series(["contains </think>" if "</think>" in str(r.get("forced_prefix_text", "")) else
                     ("empty" if not str(r.get("forced_prefix_text", "")).strip() else "donor stopped early") for r in short]).value_counts().to_dict()
    idx = [int(r["dataset_index"]) for r in recs]
    st = np.array([tb(r["instruction_following"]) for r in recs])
    off = np.array([tb(r.get("official_instruction_following", r["instruction_following"])) for r in recs])
    con = np.array([tb(r.get("continuation_instruction_following", r["instruction_following"])) for r in recs])
    row = {"Run": name, "Short prefixes (<10 tokens)": len(short), "Why": why, "IFS stored %": round(100 * st.mean(), 1),
           "IFS prefix+continuation %": round(100 * off.mean(), 1), "IFS continuation only %": round(100 * con.mean(), 1)}
    for ref_name, ref in (("Base", base), ("SFT", sft)):
        a = np.array([ref[i] for i in idx])
        d = st.astype(int) - a.astype(int)
        lo, hi = ci(d)
        b01, b10, p = mcnemar(a, st)
        row[f"vs {ref_name} (stored)"] = f"{100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}], +{b01}/-{b10}, p={p:.3g}"
        d2 = off.astype(int) - a.astype(int)
        lo, hi = ci(d2)
        row[f"vs {ref_name} (prefix+cont)"] = f"{100 * d2.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        d3 = con.astype(int) - a.astype(int)
        lo, hi = ci(d3)
        b01, b10, p = mcnemar(a, con)
        row[f"vs {ref_name} (continuation only, as reported)"] = f"{100 * d3.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}], +{b01}/-{b10}, p={p:.3g}"
    rows.append(row)
print(pd.DataFrame(rows).T.to_string(header=False))

# ------------------------------------------------------------------ D
section("D. Crossed CoT-Control runs (Qwen): early </think>, unfinished continuations, recipient effect on clean items")
cross = {}
for cond, tag in P.QWEN_TAGS.items():
    df = pd.read_csv(glob.glob(str(RES / "haskins_500" / "qwen3_14b" / f"*{tag}__*" / "results.csv"))[0]).fillna("")
    df["q"] = df.prompt_idx.astype(int)
    df["early_close"] = df.injected_prefix_text.str.contains("</think>", regex=False)
    df["unfinished"] = df.extraction_status.astype(str) != "closed"
    df["score"] = [primary_score(t, float(c)) for t, c in zip(df.task, df.continuation_compliance)]
    cross[cond] = df
print(pd.DataFrame({P.COND_LABEL[c]: {"prefix contains </think>": int(d.early_close.sum()),
                                       "unfinished (status != closed)": int(d.unfinished.sum()),
                                       "unfinished and prefix contains </think>": int((d.unfinished & d.early_close).sum()),
                                       "hit token limit": int(d.hit_token_limit.astype(str).str.lower().eq("true").sum()),
                                       "extraction_status values": d.extraction_status.value_counts().to_dict()}
                    for c, d in cross.items()}).to_string())
m = cross["SB"].merge(cross["SS"], on=["task", "q"], suffixes=("_sb", "_ss"))
for label, keep in (("all 500", np.ones(len(m), bool)),
                    ("neither prefix contains </think> and both closed", ~(m.early_close_sb | m.early_close_ss | m.unfinished_sb | m.unfinished_ss))):
    mm = m[keep]
    d = mm.score_ss - mm.score_sb
    lo, hi = ci(pd.Series(d.to_numpy()).groupby(mm.q.values).mean())
    print(f"SFT->SFT minus SFT->Base, {label}: n={len(mm)}, {100 * mm.score_sb.mean():.2f} vs {100 * mm.score_ss.mean():.2f}, "
          f"diff {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")

# ------------------------------------------------------------------ E
section("E. Fixed-opening controls (Qwen, 8 style tasks): continuation words and equal 100-word window")
ctrl = {"Base donor": cross["BB"], "SFT donor": cross["SB"]}
for f in sorted(glob.glob(str(RES / "haskins_500" / "qwen3_14b" / "*control-*" / "results.csv"))):
    d = pd.read_csv(f).fillna("")
    d["q"] = d.prompt_idx.astype(int)
    d["score"] = [primary_score(t, float(c)) for t, c in zip(d.task, d.continuation_compliance)]
    ctrl[d.control_condition.iloc[0]] = d
ref = ctrl["SFT donor"]
rows = []
for name, d in ctrl.items():
    d = d[d.task.isin(P.STYLE_TASKS)].copy()
    d["words"] = d.continuation_reasoning.map(lambda t: len(str(t).split()))
    row = {"Opening": name, "N": len(d), "Mean %": round(100 * d.score.mean(), 2), "Median continuation words": int(d.words.median())}
    m = d.merge(ref[ref.task.isin(P.STYLE_TASKS)].assign(words=lambda x: x.continuation_reasoning.map(lambda t: len(str(t).split()))),
                on=["task", "q"], suffixes=("", "_ref"))
    keep = m[(m.words >= 100) & (m.words_ref >= 100)]
    s = np.array([primary_score(t, grade_paper_protocol(t, q, P.truncate_words(str(x), 100))) for t, q, x in zip(keep.task, keep.q, keep.continuation_reasoning)])
    r = np.array([primary_score(t, grade_paper_protocol(t, q, P.truncate_words(str(x), 100))) for t, q, x in zip(keep.task, keep.q, keep.continuation_reasoning_ref)])
    lo, hi = ci(pd.Series(s - r).groupby(keep.q.values).mean())
    row["First 100 words: n, opening vs SFT donor"] = f"n={len(keep)}, {100 * s.mean():.2f} vs {100 * r.mean():.2f}, diff {100 * (s - r).mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ F
section("F. ReasonIF all-capitals items: uppercase share of letters in the reasoning")
for name in ("Qwen3-14B Base", "Qwen3-14B SFT seed 42"):
    recs = [r for r in R[name][1] if r["constraint"] == "change_case:english_capital"]
    share = []
    for r in recs:
        letters = [c for c in str(r.get("reasoning_content") or "") if c.isalpha()]
        share.append(sum(c.isupper() for c in letters) / len(letters) if letters else np.nan)
    share = np.array(share)
    print(f"{name}: n={len(recs)}, median uppercase share {np.nanmedian(share):.3f}, >=0.95: {(share >= 0.95).sum()}, >=0.99: {(share >= 0.99).sum()}, "
          f"=1.0: {(share == 1).sum()}, passes {sum(bool(r['instruction_following']) for r in recs)}")

# ------------------------------------------------------------------ G
section("G. Opening names the constraint vs CoT-Control gain, across runs and tasks (header-stripped scoring)")
HS = h_runs()
for k in [k for k in H if k.startswith("OLMo") or k.startswith("Phi")]:
    HS[k] = H[k]
HS = {k: rescore(v) for k, v in HS.items()}
NAMES = P.NAMES_CONSTRAINT
pairs_g = [("Base", k) for k in HS if k.startswith("SFT")] + [("Phi-4-reasoning Base", "Phi-4-reasoning SFT")] + \
          [("OLMo-3-7B-Think Base", k) for k in HS if k.startswith("OLMo-3-7B-Think SFT")]
rows = []
for b, s in pairs_g:
    B, S = HS[b], HS[s]
    for task, pat in NAMES.items():
        sb, ss = B[B.task == task], S[S.task == task]
        nb = np.mean([bool(re.search(pat, str(x)[:300], re.I)) for x in sb.reasoning])
        ns = np.mean([bool(re.search(pat, str(x)[:300], re.I)) for x in ss.reasoning])
        rows.append({"run": s, "task": task, "names_rate_change": ns - nb, "gain": 100 * (ss.primary_s.mean() - sb.primary_s.mean())})
g = pd.DataFrame(rows)
from scipy.stats import spearmanr  # noqa: E402
rho, p = spearmanr(g.names_rate_change, g.gain)
print(f"run x task cells: {len(g)}; Spearman rho(names-rate change, gain) = {rho:.2f} (p={p:.3g})")
print(g.pivot(index="task", columns="run", values="gain").round(1).to_string())
print(g.pivot(index="task", columns="run", values="names_rate_change").round(2).to_string())
for task in NAMES:
    gg = g[g.task == task]
    if gg.names_rate_change.std() > 0:
        r_, p_ = spearmanr(gg.names_rate_change, gg.gain)
        print(f"  {task:22s} across {len(gg)} runs: rho={r_:.2f}")

# ------------------------------------------------------------------ H
section("H. Training data: does the opening restate the request?")
data = json.loads((RES.parent / "data/training/gpt52_high_reasoning_original.json").read_text(encoding="utf-8"))
STOP = set("a an the and or of for to in on with by as at from into via is are be your you me my i we our it its this that these those "
           "plus include including e g etc how what design build create help plan explain set up lay out".split())


def content_words(t):
    return {w for w in re.findall(r"[a-z][a-z0-9\-]+", t.lower()) if w not in STOP and len(w) > 2}


def opening(ex):
    think = re.search(r"<think>(.*?)</think>", ex["output"], re.S).group(1)
    head = re.match(r"\s*\*\*([^*\n]{1,120})\*\*", think)
    rest = think[head.end():] if head else think
    first = re.split(r"(?<=[.!?])\s|\n", rest.strip(), maxsplit=1)[0]
    return (head.group(1) if head else ""), first


rng = np.random.default_rng(42)
own, shuf, head_own, verbs = [], [], [], 0
perm = rng.permutation(len(data))
for i, ex in enumerate(data):
    h, first = opening(ex)
    q = content_words(ex["instruction"])
    own.append(len(q & content_words(first)) / max(1, len(q)))
    shuf.append(len(content_words(data[perm[i]]["instruction"]) & content_words(first)) / max(1, len(content_words(data[perm[i]]["instruction"]))))
    head_own.append(len(q & content_words(h)) / max(1, len(q)))
    verbs += bool(re.match(r"\s*(I need to|I'm|I am|I'll|I will|I should|I want to|The user (wants|is asking|asks))", first))
print(f"first sentence covers {100 * np.mean(own):.1f}% of the request's content words (another example's request: {100 * np.mean(shuf):.1f}%)")
print(f"heading covers {100 * np.mean(head_own):.1f}% of the request's content words")
print(f"first sentence opens with 'I need to / I'm / I'll / The user wants...': {verbs}/{len(data)}")
print("examples:", [opening(ex) for ex in data[:3]])

# ------------------------------------------------------------------ I
section("I. KL caches (Qwen3-14B, CoT-Control): first-position KL by whether the SFT trace opens with a bold heading")
cache = json.loads((RES / "kl_divergence/caches_float32/kl_cache_qwen3_14b_haskins_500.json").read_text(encoding="utf-8"))
sft_recs = [json.loads(l) for l in (RES / "haskins_500/qwen3_14b/haskins_qwen3_14b_vllm_calibrated_sft_31pct.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
recs = cache["records"]
keys = list(recs)
by_key = {f'{r["task"]}:{int(r["prompt_idx"])}': r for r in sft_recs}
assert all(k in by_key for k in keys), [k for k in keys if k not in by_key][:5]
rows = []
for k in keys:
    r = by_key[k]
    kl = json.loads(recs[k]["reasoning_kl"]) if isinstance(recs[k]["reasoning_kl"], str) else recs[k]["reasoning_kl"]
    rows.append({"heading": bool(LEAD.match(r["reasoning"] or "")), "kl1": kl[0] if kl else np.nan,
                 "kl2_10": float(np.mean(kl[1:10])) if len(kl) > 1 else np.nan, "kl11_100": float(np.mean(kl[10:100])) if len(kl) > 10 else np.nan})
k_ = pd.DataFrame(rows)
print("cache keys (task:question) matched to SFT records:", len(keys))
print(k_.groupby("heading")[["kl1", "kl2_10", "kl11_100"]].agg(["mean", "count"]).round(2).to_string())
