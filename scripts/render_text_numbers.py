"""Fill number placeholders in a text template from paper_tables/numbers7.json (so no number in the revised text is
typed by hand). [[key]] -> value; [[key|est]] -> $+x$; [[key|ci]] -> $+x$, $[lo, hi]$; [[key|n]] -> pair count;
[[min:k1,k2,...]] / [[max:k1,k2,...]] -> smallest / largest value. Fails on a missing key.

  python scripts/render_text_numbers.py <template> <output>
"""
import json
import re
import sys
from pathlib import Path

NUM = json.loads((Path(__file__).resolve().parents[1] / "paper_tables" / "numbers7.json").read_text(encoding="utf-8"))


def fmt(x, nd=2):
    s = f"{x:.{nd}f}"
    return s


def sg(x):
    return f"+{x:.2f}" if x >= 0 else f"-{abs(x):.2f}"


def val(key):
    v = NUM[key]
    return v["est"] if isinstance(v, dict) else v


def sub(m):
    body = m.group(1)
    if body.startswith(("min:", "max:")):
        vals = [val(k) for k in body[4:].split(",")]
        x = min(vals) if body.startswith("min:") else max(vals)
        nd = 1 if body[4:].startswith(("rifIFS_", "rifACC_")) else 2
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    key, _, mode = body.partition("|")
    v = NUM[key]
    if not mode:
        if isinstance(v, dict):
            raise KeyError(f"{key} needs a mode")
        return f"{v:.1f}" if isinstance(v, float) and key.startswith(("lower_", "task_", "rif_", "rifIFS_", "rifACC_", "open_rest", "open_selfheaders_lower_after")) else (f"{v:.2f}" if isinstance(v, float) else str(v))
    if mode == "est":
        return f"${sg(v['est'])}$"
    if mode == "ci":
        return f"${sg(v['est'])}$, $[{fmt(v['lo'])}, {fmt(v['hi'])}]$"
    if mode == "ci1":
        return f"${'+' if v['est'] >= 0 else '-'}{abs(v['est']):.1f}$, $[{v['lo']:.1f}, {v['hi']:.1f}]$"
    if mode == "num":
        return f"{v['est']:.2f}"
    if mode == "n":
        return str(v["n"])
    raise ValueError(mode)


src, dst = sys.argv[1], sys.argv[2]
text = Path(src).read_text(encoding="utf-8")
out = re.sub(r"\[\[([^\]]+)\]\]", sub, text)
Path(dst).write_text(out, encoding="utf-8")
print("rendered", dst, len(re.findall(r"\[\[", text)), "placeholders")
