"""Does the opening heading itself obey the constraint, and does the rest of the reasoning follow it?

For the case tasks (uppercase, lowercase, alternating case) and third person, a heading "obeys" if its letters are all
uppercase / all lowercase / alternate in case, or (third person) it contains no first-person pronoun. For every run
with headings, prints how often the heading obeys and the mean compliance of the reasoning *after* the heading
(graded with the heading removed), split by whether the heading obeyed.

  python scripts/heading_obeys_rule.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from haskins_header_stripped import rescore, runs  # noqa: E402

LEAD = re.compile(r"^\s*\*\*([^*\n]{1,120})\*\*")
FIRST = re.compile(r"\b(I|I'm|I'll|I've|I'd|me|my|mine|myself)\b", re.I)


def obeys(task, h):
    letters = [c for c in h if c.isalpha()]
    if not letters:
        return None
    if task == "uppercase_thinking":
        return all(c.isupper() for c in letters)
    if task == "lowercase_thinking":
        return all(c.islower() for c in letters)
    if task == "alternating_case":
        return all(a.isupper() != b.isupper() for w in h.split() for a, b in zip([c for c in w if c.isalpha()], [c for c in w if c.isalpha()][1:]))
    if task == "third_person":
        return not FIRST.search(h)
    return None


H = {k: rescore(v) for k, v in runs().items()}
rows = []
for name, d in H.items():
    for task in ("uppercase_thinking", "lowercase_thinking", "alternating_case", "third_person"):
        g = d[d.task == task]
        heads = [LEAD.match(str(x)) for x in g.reasoning]
        ob = np.array([obeys(task, m.group(1)) if m else None for m in heads], dtype=object)
        has = np.array([m is not None for m in heads])
        if has.sum() == 0:
            continue
        o = np.array([bool(x) for x in ob[has]])
        comp = g.primary_s.to_numpy()[has]
        rows.append({"run": name, "task": task, "headings": int(has.sum()), "heading obeys": int(o.sum()),
                     "compliance after obeying heading": round(100 * comp[o].mean(), 1) if o.any() else np.nan,
                     "compliance after other heading": round(100 * comp[~o].mean(), 1) if (~o).any() else np.nan})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
