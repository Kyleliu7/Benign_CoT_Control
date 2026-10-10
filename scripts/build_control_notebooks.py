"""
Derive two new evaluation notebooks from notebooks/Haskins_500_vLLM_2x2_SFT_Donor_to_Base_Evaluation.ipynb by small, asserted text patches.

1. Haskins_500_vLLM_Prefix_Controls_Evaluation.ipynb
   Base recipient continues from a FIXED opening read from data/prefix_controls/prefix_controls.json instead of a generated donor opening.
   Set CONTROL_CONDITION in the config cell to: handcrafted | named | mismatched | cross_question (run the notebook once per condition).

2. Haskins_500_vLLM_2x2_SFT_Donor_to_Base_PrefixLength_Evaluation.ipynb
   The original SFT-donor -> base-recipient run with a longer donor opening (weight switch after N tokens). Set PREFIX_TOKENS to 30 or 100.
   (Only ~97% / ~91% of Qwen SFT reasoning traces are at least 30 / 100 tokens long; longer N mostly measures traces that already finished.)

Every patch asserts that the text it replaces exists, so a changed source notebook fails loudly. Outputs are cleared.
Usage: python scripts/build_control_notebooks.py
"""
import ast
import copy
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "notebooks" / "Haskins_500_vLLM_2x2_SFT_Donor_to_Base_Evaluation.ipynb"


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def text(nb, i):
    return "".join(nb["cells"][i]["source"])


def put(nb, i, s):
    nb["cells"][i]["source"] = s.splitlines(keepends=True)


def sub(s, old, new, count=1):
    assert old in s, f"patch target not found: {old[:70]!r}"
    return s.replace(old, new, count)


def clear_outputs(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None


def check_syntax(nb, name):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            ast.parse(src)
        except SyntaxError as exc:
            raise SystemExit(f"{name}: cell {i} does not parse: {exc}")


CONTROL_CONFIG = '''
# ---- PREFIX-CONTROL SETTINGS --------------------------------------------------------------------
import hashlib as _hashlib
import json
CONTROL_CONDITION = "handcrafted"   # handcrafted | named | mismatched | cross_question | modal_sft
CONTROL_FILE_CANDIDATES = [Path("prefix_controls.json"), Path("data/prefix_controls/prefix_controls.json"),
                           Path("../data/prefix_controls/prefix_controls.json")]
CONTROL_FILE = next((p for p in CONTROL_FILE_CANDIDATES if p.exists()), None)
assert CONTROL_FILE is not None, "prefix_controls.json not found: copy data/prefix_controls/prefix_controls.json next to this notebook"
CONTROL_FILE_SHA256 = _hashlib.sha256(CONTROL_FILE.read_bytes()).hexdigest()
CONTROL_PAYLOAD = json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
assert CONTROL_CONDITION in CONTROL_PAYLOAD["conditions"], CONTROL_CONDITION
assert PREFIX_TOKENS == CONTROL_PAYLOAD["prefix_tokens_including_think"] == 10   # fixed texts may be shorter than 10 tokens
'''

CONTROL_PHASE1 = '''prefix_cache_path = run_dir / "prefix_cache.json"
# ==============================================================================
# PHASE 1 (CONTROL VERSION): FIXED OPENINGS FROM prefix_controls.json -- NO DONOR GENERATION
# ==============================================================================
think_open_id = tokenizer.convert_tokens_to_ids("<think>")
assert isinstance(think_open_id, int) and think_open_id >= 0
condition_table = CONTROL_PAYLOAD["conditions"][CONTROL_CONDITION]
prefix_cache = {}
for job in JOBS:
    entry = condition_table[job["task"]]
    if isinstance(entry, dict) and str(job["prompt_idx"]) in entry:   # per-question entries (cross_question)
        entry = entry[str(job["prompt_idx"])]
    if isinstance(entry, dict) and "token_ids" in entry:
        p_ids = [int(t) for t in entry["token_ids"]]
    else:
        body = entry["text"] if isinstance(entry, dict) else entry
        body_ids = tokenizer.encode(body, add_special_tokens=False)
        assert len(body_ids) <= PREFIX_TOKENS - 1, f"opening is {len(body_ids)} tokens (max {PREFIX_TOKENS - 1}); it would cut off the rule: {body!r}"
        p_ids = [think_open_id] + body_ids
    assert len(p_ids) <= PREFIX_TOKENS and p_ids[0] == think_open_id, (job["id"], len(p_ids))
    p_text = tokenizer.decode(p_ids)
    job["prefix_text"] = p_text
    job["prefix_token_ids"] = p_ids
    job["continuation_prompt_token_ids"] = job["prompt_on_token_ids"] + p_ids
    prefix_cache[job["id"]] = {"task": job["task"], "prompt_idx": job["prompt_idx"], "prefix_text": p_text, "prefix_token_ids": p_ids,
                               "control_condition": CONTROL_CONDITION}
atomic_json(prefix_cache_path, prefix_cache)
print(f"Control condition '{CONTROL_CONDITION}': prepared {len(prefix_cache)} fixed {PREFIX_TOKENS}-token openings (file sha256 {CONTROL_FILE_SHA256[:12]})")
'''

EARLY_FINISH_WARNING = '''
_close_id = tokenizer.convert_tokens_to_ids("</think>")
_early = [j["id"] for j in JOBS if _close_id in j["prefix_token_ids"]]
print(f"{len(_early)} of {len(JOBS)} donor prefixes already contain </think> (donor finished reasoning inside the prefix window).")
if _early:
    print("WARNING: for these items the recipient continues after the donor's reasoning ended; report them separately or use a smaller PREFIX_TOKENS.")
'''


def build_controls() -> dict:
    nb = load()
    # title
    put(nb, 0, "# Haskins CoT Controllability - Prefix Controls: fixed opening -> Base Recipient\n\n"
               "The base recipient continues from a fixed opening taken from `prefix_controls.json` (conditions: handcrafted, named, mismatched, cross_question, modal_sft). "
               "No donor model generates anything. Set `CONTROL_CONDITION` in the config cell and run once per condition. "
               "All other settings are identical to the 2x2 SFT-donor -> base-recipient run.\n")
    # config
    s = text(nb, 3)
    s = sub(s, 'PREFIX_MODE = "ON"', 'PREFIX_MODE = "ON"')
    s = re.sub(r'^DONOR_MODEL = "sft".*$', 'DONOR_MODEL = "control"   # fixed openings from prefix_controls.json (no donor generation)', s, flags=re.M)
    assert 'DONOR_MODEL = "control"' in s
    s = sub(s, 'PROTOCOL_NAME = "haskins-vllm-2x2-sft-to-base-on-v1"', 'PROTOCOL_NAME = f"haskins-vllm-prefix-control-{CONTROL_CONDITION}-v1"')
    s = s.rstrip("\n") + "\n" + CONTROL_CONFIG
    # PROTOCOL_NAME uses CONTROL_CONDITION, so the control block must come first: move the PROTOCOL_NAME line after it
    s = sub(s, 'PROTOCOL_NAME = f"haskins-vllm-prefix-control-{CONTROL_CONDITION}-v1"\n', "")
    s = s.rstrip("\n") + '\nPROTOCOL_NAME = f"haskins-vllm-prefix-control-{CONTROL_CONDITION}-v1"\n'
    put(nb, 3, s)
    # run directory + fingerprint
    s = text(nb, 14)
    s = sub(s, '"prefix_mode": "ON",', '"prefix_mode": "ON",\n    "control_condition": CONTROL_CONDITION,\n    "control_file_sha256": CONTROL_FILE_SHA256,')
    s = sub(s, "__qwen__qwen3-14b__2x2-sft-to-base-on-10tok\"", "__qwen__qwen3-14b__control-{CONTROL_CONDITION}-{PREFIX_TOKENS}tok\"")
    put(nb, 14, s)
    # phase 1 -> fixed openings
    s = text(nb, 20)
    a = s.index('prefix_cache_path = run_dir / "prefix_cache.json"')
    b = s.index("# Sample prefix check")
    s = s[:a] + CONTROL_PHASE1 + "\n" + s[b:]
    s = sub(s, '"prefix_mode": PREFIX_MODE,', '"prefix_mode": PREFIX_MODE,\n        "control_condition": CONTROL_CONDITION,')
    put(nb, 20, s)
    clear_outputs(nb)
    check_syntax(nb, "controls")
    return nb


def build_prefix_length() -> dict:
    nb = load()
    put(nb, 0, "# Haskins CoT Controllability - SFT Donor (Constraint ON) -> Base Recipient with a longer donor opening (weight switch after N tokens)\n\n"
               "Identical to the 2x2 SFT-donor -> base-recipient run, but the SFT model writes the first `PREFIX_TOKENS` tokens (set to 30 or 100) "
               "before the unchanged base model continues. The first token is `<think>`.\n")
    s = text(nb, 3)
    s = sub(s, "PREFIX_TOKENS = 10     # Injected prefix token budget", "PREFIX_TOKENS = 30     # Injected prefix token budget: set to 30 or 100 (10 is the original run)")
    s = sub(s, 'PROTOCOL_NAME = "haskins-vllm-2x2-sft-to-base-on-v1"', 'PROTOCOL_NAME = f"haskins-vllm-2x2-sft-to-base-on-{PREFIX_TOKENS}tok-v1"')
    put(nb, 3, s)
    s = text(nb, 14)
    s = sub(s, "__qwen__qwen3-14b__2x2-sft-to-base-on-10tok\"", "__qwen__qwen3-14b__2x2-sft-to-base-on-{PREFIX_TOKENS}tok\"")
    put(nb, 14, s)
    s = text(nb, 20)
    s = sub(s, "# Sample prefix check", EARLY_FINISH_WARNING + "\n# Sample prefix check")
    put(nb, 20, s)
    clear_outputs(nb)
    check_syntax(nb, "prefix-length")
    return nb


def main():
    out = REPO / "notebooks"
    for name, nb in (("Haskins_500_vLLM_Prefix_Controls_Evaluation.ipynb", build_controls()),
                     ("Haskins_500_vLLM_2x2_SFT_Donor_to_Base_PrefixLength_Evaluation.ipynb", build_prefix_length())):
        (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
