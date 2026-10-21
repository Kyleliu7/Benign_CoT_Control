"""Print lowercase_thinking traces from each run for the same questions: the trace's score (share of sentences with
only lowercase letters, as graded), its first sentences, and which of them fail.

  python scripts/lowercase_examples.py [question indices, default 0 7 21]
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_followups as F  # noqa: E402
from haskins_header_stripped import runs  # noqa: E402
from lowercase_length_check import KEEP, lower_ok, sentences  # noqa: E402

QS = [int(x) for x in sys.argv[1:]] or [0, 7, 21]
H = runs()
f = glob.glob(str(F.RES / "haskins_500/qwen3_14b_training_variants/self_distill/haskins_500/qwen3_14b/*/results.csv"))
if f:
    H["SFT self-distillation"] = F._prep(pd.read_csv(f[0]))
for q in QS:
    print("=" * 100, f"\nquestion {q}")
    for k in KEEP:
        if k not in H:
            continue
        d = H[k]
        x = d[(d.task == "lowercase_thinking") & (d.q == q)].reasoning
        if not len(x):
            continue
        ss = sentences(x.iloc[0])
        ok = [lower_ok(s) for s in ss]
        print(f"\n--- {k}: {sum(ok)}/{len(ss)} sentences lowercase = {100 * sum(ok) / max(len(ss), 1):.1f}%")
        for s, o in list(zip(ss, ok))[:6]:
            print(f"  [{'ok ' if o else 'BAD'}] {s[:150]}")
