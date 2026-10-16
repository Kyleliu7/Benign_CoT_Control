"""Teacher-controllability test: two training sets on the same 212 s1K questions, with raw reasoning from a teacher of
high CoT controllability (Claude 3.7 Sonnet, 51.1% on stylistic instructions in Chen et al. 2026, Table 2) and one of
very low controllability (DeepSeek R1, 0.2%).

Sources (MIT): simplescaling/s1K-claude-3-7-sonnet (claude_thinking_trajectory, claude_attempt) and
simplescaling/s1K-1.1 (deepseek_thinking_trajectory, deepseek_attempt); both contain the same 1,000 s1K questions.
Each example is formatted like data/training/gpt52_high_reasoning_original.json: instruction = question,
output = "<think>" + reasoning + "</think>\\n" + answer. A question is eligible if (a) both teachers' examples fit the
training cutoff (Qwen3-14B chat template + output <= CUTOFF tokens) and (b) the question shares no 8-gram with any of
the 50 CoT-Control questions. 212 eligible questions are drawn with random.Random(42). Writes
data/training/s1k_claude37_212.json, data/training/s1k_deepseek_r1_212.json, data/training/s1k_selection.json, and
registers both sets in data/training/dataset_info.json.

  python scripts/make_s1k_teacher_datasets.py [CUTOFF]      (default 8192)
"""
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data" / "training"
CUTOFF = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
N = 212


def grams(t, n=8):
    w = re.findall(r"\w+", t.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    c = load_dataset("simplescaling/s1K-claude-3-7-sonnet", split="train")
    r = load_dataset("simplescaling/s1K-1.1", split="train")
    rq = {q: i for i, q in enumerate(r["question"])}
    cot_q = [p["prompt"] for p in json.loads((REPO / "data/eval_prompts/diverse_prompts.json").read_text(encoding="utf-8"))]
    cot_grams = set().union(*(grams(q) for q in cot_q))

    def fmt(q, think, ans):
        return {"instruction": q, "input": "", "output": "<think>" + (think or "").strip() + "</think>\n" + (ans or "").strip()}

    def n_tokens(ex):
        prompt = tok.apply_chat_template([{"role": "user", "content": ex["instruction"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        return len(tok(prompt + ex["output"], add_special_tokens=False)["input_ids"])

    stats = {"questions": len(c), "matched": 0, "missing_reasoning": 0, "too_long": 0, "overlap_with_cot_control": 0}
    eligible = []
    for i, row in enumerate(c):
        j = rq.get(row["question"])
        if j is None:
            continue
        stats["matched"] += 1
        rr = r[j]
        if not (row["claude_thinking_trajectory"] and row["claude_attempt"] and rr["deepseek_thinking_trajectory"] and rr["deepseek_attempt"]):
            stats["missing_reasoning"] += 1
            continue
        if grams(row["question"]) & cot_grams:
            stats["overlap_with_cot_control"] += 1
            continue
        ec = fmt(row["question"], row["claude_thinking_trajectory"], row["claude_attempt"])
        er = fmt(row["question"], rr["deepseek_thinking_trajectory"], rr["deepseek_attempt"])
        lc, lr = n_tokens(ec), n_tokens(er)
        if max(lc, lr) > CUTOFF:
            stats["too_long"] += 1
            continue
        eligible.append((i, ec, er, lc, lr))
    stats["eligible"] = len(eligible)
    assert len(eligible) >= N, stats
    pick = sorted(random.Random(42).sample(range(len(eligible)), N))
    sel = [eligible[k] for k in pick]
    (D / "s1k_claude37_212.json").write_text(json.dumps([e[1] for e in sel], ensure_ascii=False, indent=2), encoding="utf-8")
    (D / "s1k_deepseek_r1_212.json").write_text(json.dumps([e[2] for e in sel], ensure_ascii=False, indent=2), encoding="utf-8")
    import statistics as st
    summary = {"cutoff_tokens": CUTOFF, "n": N, "seed": 42, "stats": stats, "s1k_indices": [e[0] for e in sel],
               "median_tokens": {"claude37": st.median(e[3] for e in sel), "deepseek_r1": st.median(e[4] for e in sel)},
               "median_reasoning_words": {"claude37": st.median(len(re.search(r"<think>(.*?)</think>", e[1]["output"], re.S).group(1).split()) for e in sel),
                                          "deepseek_r1": st.median(len(re.search(r"<think>(.*?)</think>", e[2]["output"], re.S).group(1).split()) for e in sel)}}
    (D / "s1k_selection.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    info = json.loads((D / "dataset_info.json").read_text(encoding="utf-8"))
    for name in ("s1k_claude37_212", "s1k_deepseek_r1_212"):
        info[name] = {**info["gpt52_high_reasoning_original"], "file_name": name + ".json"}
    (D / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "s1k_indices"}, indent=1))
    print("first claude opening:", sel[0][1]["output"][:160].replace("\n", " "))
    print("first r1 opening:", sel[0][2]["output"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
