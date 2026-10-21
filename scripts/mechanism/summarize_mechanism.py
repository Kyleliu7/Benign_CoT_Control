"""Summaries of results/mechanism/ (written by the scripts in this folder).

  surprisal  per model family: mean surprisal (negative log-probability, nats per token) of each segment of the
             training targets under the base model and each adapter, and the reduction (base minus adapter) with a
             95% bootstrap interval over examples, separately for training and validation examples.
  attention  share of the first reasoning position's attention on the constraint instruction and on the question,
             base vs adapters, by layer band; probability that the first token is "**"; patching gap recovered.
  openings   bold heading and constraint naming in sampled openings.

  python scripts/mechanism/summarize_mechanism.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "mechanism"
SEGS = ["lead_header", "first_sentence", "other_headers", "body", "answer"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
sys.path.insert(0, str(REPO / "scripts"))


def boot(v, n=10_000, seed=42):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def surprisal(path):
    d = json.loads(path.read_text())
    rows = pd.DataFrame(d["rows"])
    labels = ["base"] + list(d["meta"]["adapters"])
    out = []
    for split in ("train", "val"):
        r = rows[rows.split == split]
        for seg in SEGS:
            n = r[f"base__{seg}__n"].to_numpy()
            keep = n > 0
            row = {"split": split, "segment": seg, "examples": int(keep.sum()), "tokens": int(n[keep].sum())}
            base_nll = -r[f"base__{seg}__sum"].to_numpy()[keep] / n[keep]
            row["base nll"] = round(float(base_nll.mean()), 3)
            for lab in labels[1:]:
                ad = -r[f"{lab}__{seg}__sum"].to_numpy()[keep] / n[keep]
                red = base_nll - ad
                lo, hi = boot(red)
                row[f"{lab} nll"] = round(float(ad.mean()), 3)
                row[f"{lab} reduction [95% CI]"] = f"{red.mean():.3f} [{lo:.3f}, {hi:.3f}]"
            out.append(row)
    return d["meta"], pd.DataFrame(out)


def attention(folder):
    z = np.load(folder / "attention_first_position.npz")
    meta = json.loads((folder / "attention_first_position_items.json").read_text())
    items = pd.DataFrame(meta["items"])
    labels = sorted({k.split("__")[0] for k in z.files})
    labels = ["base"] + [l for l in labels if l != "base"]
    L = z["base__constraint"].shape[1]
    bands = {"layers 0-13": range(0, L // 3), "14-26": range(L // 3, 2 * L // 3), f"27-{L - 1}": range(2 * L // 3, L)}
    rows = []
    for lab in labels:
        c, q = z[f"{lab}__constraint"], z[f"{lab}__question"]  # [N, L, H]
        row = {"model": lab, "P(first token = **)": round(float(z[f"{lab}__p_star"].mean()), 3)}
        for bname, ls in bands.items():
            row[f"constraint share, {bname}"] = round(float(c[:, list(ls)].mean()), 4)
            row[f"question share, {bname}"] = round(float(q[:, list(ls)].mean()), 4)
        rows.append(row)
    tab = pd.DataFrame(rows)
    diffs = []
    qidx = items.q.to_numpy()
    for lab in labels[1:]:
        for bname, ls in bands.items():
            d = (z[f"{lab}__constraint"][:, list(ls)] - z["base__constraint"][:, list(ls)]).mean(axis=(1, 2))
            lo, hi = boot(pd.Series(d).groupby(qidx).mean().to_numpy())
            diffs.append({"model": lab, "band": bname, "constraint share change": f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"})
        # per-layer top heads
    per_layer = pd.DataFrame({lab: (z[f"{lab}__constraint"] - z["base__constraint"]).mean(axis=(0, 2)) for lab in labels[1:]})
    top = []
    for lab in labels[1:]:
        dh = (z[f"{lab}__constraint"] - z["base__constraint"]).mean(0)  # [L, H]
        for flat in np.argsort(dh.ravel())[::-1][:5]:
            top.append({"model": lab, "layer": int(flat // dh.shape[1]), "head": int(flat % dh.shape[1]), "change": round(float(dh.ravel()[flat]), 4)})
    by_task = pd.DataFrame({lab: pd.Series((z[f"{lab}__constraint"] - z["base__constraint"]).mean(axis=(1, 2))).groupby(items.task.to_numpy()).mean()
                            for lab in labels[1:]})
    out = {"table": tab, "diffs": pd.DataFrame(diffs), "per_layer": per_layer, "top_heads": pd.DataFrame(top), "by_task": by_task,
           "top1": {lab: pd.Series(meta["top1"][lab]).value_counts().head(5).to_dict() for lab in labels}}
    pf = folder / "attention_patching.json"
    if pf.exists():
        p = pd.DataFrame(json.loads(pf.read_text())["rows"])
        gap = (p.sft - p.base).sum()
        prow = [{"condition": "sanity: base with its own attention", "mean logp(opening)": round(p.base_self_patch.mean(), 3),
                 "vs base": round((p.base_self_patch - p.base).abs().max(), 5)}]
        for c in [c for c in p.columns if c.startswith("base_with_sft_attn_")]:
            lo, hi = boot(pd.Series(((p[c] - p.base) / max(1e-9, (p.sft - p.base).mean())).to_numpy()).groupby(p.q.to_numpy()).mean())
            prow.append({"condition": c, "mean logp(opening)": round(p[c].mean(), 3), "gap recovered": round(float((p[c] - p.base).sum() / gap), 3)})
        for c in [c for c in p.columns if c.startswith("sft_with_base_attn_")]:
            prow.append({"condition": c, "mean logp(opening)": round(p[c].mean(), 3), "gap lost": round(float((p.sft - p[c]).sum() / gap), 3)})
        out["patching"] = pd.DataFrame(prow)
        out["patching_n"] = len(p)
        out["patching_means"] = {"base": round(p.base.mean(), 3), "sft": round(p.sft.mean(), 3)}
    return out


def openings(path):
    import analysis_prefix_length_paired as P
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    d = pd.DataFrame(rows)
    body = d.opening.str.replace(r"^\s*<think>\s*", "", regex=True)
    d["heading"] = body.str.match(r"\s*\*\*")
    d["names"] = [bool(re.search(P.NAMES_CONSTRAINT.get(t, r"$^"), b[:300], re.I)) for t, b in zip(d.task, body)]
    d["think_open"] = d.opening.str.contains("<think>", regex=False)
    return d, body


if __name__ == "__main__":
    for f in sorted(OUT.glob("surprisal_*.json")):
        meta, t = surprisal(f)
        print(f"\n== {f.name}: base {meta['base']}, adapters {list(meta['adapters'])}, n={meta['n']} (val {meta['n_val']})")
        print(t.to_string(index=False))
    for folder in sorted(OUT.glob("attention_*")):
        if not folder.is_dir():
            continue
        a = attention(folder)
        print(f"\n== {folder.name}")
        for k in ("table", "diffs", "top_heads", "by_task"):
            print(f"-- {k}")
            print(a[k].to_string())
        print("-- per-layer change in constraint share (adapter - base), mean over items and heads")
        print(a["per_layer"].round(4).T.to_string())
        print("-- top-1 first token:", a["top1"])
        if "patching" in a:
            print(f"-- patching (n={a['patching_n']} openings, mean log p base/sft: {a['patching_means']})")
            print(a["patching"].to_string(index=False))
    for f in sorted(OUT.glob("openings_*.jsonl")):
        d, body = openings(f)
        print(f"\n== {f.name}: n={len(d)}, opens with bold heading {int(d.heading.sum())}, names the constraint early {int(d.names.sum())}, "
              f"contains <think> {int(d.think_open.sum())}")
        print(d.groupby("task")[["heading", "names"]].sum().T.to_string())
        for t in ("lowercase_thinking", "third_person", "uppercase_thinking"):
            ex = body[d.task == t].head(2).tolist()
            print(f"  {t}: " + " || ".join(repr(x[:140]) for x in ex))
