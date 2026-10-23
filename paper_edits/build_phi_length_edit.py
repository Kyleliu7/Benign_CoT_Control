"""Phi-4-reasoning length sentence for the arXiv paper (Kyle: clarify that the gain can't be attributed to length either).
Numbers from results/summary_tables/length_overview.txt (scripts/length_overview.py). Writes paper_edits/phi_length_edit.json [old, new]."""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LO = (REPO / "results/summary_tables/length_overview.txt").read_text(encoding="utf-8")
r_sft = re.search(r"Phi-4-reasoning SFT: correlation of log reasoning tokens with seven-task compliance r = ([+-][\d.]+)", LO).group(1)
r_base = re.search(r"Phi-4-reasoning Base: correlation of log reasoning tokens with seven-task compliance r = ([+-][\d.]+)", LO).group(1)
m = re.search(r"^   SFT: unmatched gain [^;]+; length-matched within task ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\] \(n=(\d+)\)", LO, re.M)
est, lo, hi, n = float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
old = ("and too few traces can be matched on length to test further. The gain cannot be separated from length but cannot be solely "
       "attributed to it either.")
new = (f"and only {n} traces can be matched on length within a task. The gain therefore cannot be separated from length, but the data do not "
       f"show that length causes it either: within each Phi run, shorter traces are not more compliant (correlation of log reasoning length "
       f"with compliance $r = {float(r_sft):+.2f}$ for the fine-tuned model and ${float(r_base):+.2f}$ for Base), the matched traces lean toward "
       f"a gain (${est:+.1f}$, $[{lo:.1f}, {hi:.1f}]$), and Phi's s1K students keep their gains at matched length (below).")
(REPO / "paper_edits/phi_length_edit.json").write_text(json.dumps([old, new], ensure_ascii=True), encoding="utf-8")
print(new)
