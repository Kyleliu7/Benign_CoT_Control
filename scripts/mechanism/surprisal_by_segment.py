"""What did fine-tuning learn from the training data? Per-token log-probability of the 212 GPT-5.2 training targets
under a base model and its LoRA adapters, grouped by segment of the target.

Segments of each target (data/training/gpt52_high_reasoning_original.json):
  lead_header     the bold heading that opens the reasoning (**...**)
  first_sentence  the first sentence after it (up to . ! ? or a line break), which usually restates the request
  other_headers   later bold headings
  body            the rest of the reasoning
  answer          everything after </think>
The model input is the base model's chat-template generation prompt for the instruction followed by the target text
exactly as stored (identical for base and adapters). Only target tokens are scored; a token belongs to the segment
that contains its first character. Examples are split into the 190 training and 22 validation examples with the same
call LlamaFactory uses (train_test_split, test_size=0.1, seed=42).

  python surprisal_by_segment.py --base Qwen/Qwen3-14B --adapter sft42=<path> ... --out <file.json>
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/training/gpt52_high_reasoning_original.json"
HEAD = re.compile(r"\*\*[^*\n]{1,120}\*\*")
SEGS = ["lead_header", "first_sentence", "other_headers", "body", "answer"]


def segment_labels(output: str):
    """Character-level labels for the target string."""
    lab = np.array(["other"] * len(output), dtype=object)
    m = re.search(r"<think>(.*?)</think>", output, re.S)
    t0, t1 = m.start(1), m.end(1)
    lab[t0:t1] = "body"
    lab[m.end():] = "answer"
    heads = [(t0 + h.start(), t0 + h.end()) for h in HEAD.finditer(output[t0:t1])]
    for i, (a, b) in enumerate(heads):
        lead = i == 0 and output[t0:a].strip() == ""
        lab[a:b] = "lead_header" if lead else "other_headers"
        if lead:
            rest = output[b:t1]
            s = len(rest) - len(rest.lstrip())
            e = re.search(r"[.!?](\s|$)|\n", rest[s:])
            end = b + s + (e.end() if e else len(rest) - s)
            nxt = [h for h in heads[1:] if h[0] < end]
            end = min([end] + [h[0] for h in nxt])
            lab[b + s:end] = "first_sentence"
    return lab


def val_indices(n):
    from datasets import Dataset
    sp = Dataset.from_dict({"i": list(range(n))}).train_test_split(test_size=0.1, seed=42)
    return set(sp["test"]["i"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", action="append", default=[], help="name=path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    data = json.loads(DATA.read_text(encoding="utf-8"))
    val = val_indices(len(data))
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    adapters = dict(a.split("=", 1) for a in args.adapter)
    for i, (name, path) in enumerate(adapters.items()):
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
    model.eval()
    labels = ["base"] + list(adapters)
    rows = []
    for idx, ex in enumerate(data):
        prompt = tok.apply_chat_template([{"role": "user", "content": ex["instruction"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        target = ex["output"]
        if prompt.rstrip().endswith("<think>") and target.startswith("<think>"):
            target = target[len("<think>"):]
        text = prompt + target
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        lab_chars = segment_labels(ex["output"]) if target is ex["output"] else segment_labels("<think>" + target)[len("<think>"):]
        p0 = len(prompt)
        tok_lab = [(lab_chars[s - p0] if s >= p0 and s - p0 < len(lab_chars) else "prompt") for s, _ in offs]
        x = torch.tensor([ids], device="cuda")
        row = {"idx": idx, "split": "val" if idx in val else "train", "n_tokens": len(ids)}
        for lab in labels:
            if lab == "base":
                ctx = model.disable_adapter() if adapters else torch.no_grad()
            else:
                model.set_adapter(lab)
                ctx = torch.no_grad()
            with ctx, torch.no_grad():
                logits = model(x).logits[0]
            lp = np.zeros(len(ids))
            for a in range(0, len(ids) - 1, 1024):
                b = min(a + 1024, len(ids) - 1)
                l = torch.log_softmax(logits[a:b].float(), -1)
                lp[a + 1:b + 1] = l.gather(1, x[0, a + 1:b + 1, None]).squeeze(1).cpu().numpy()
            for seg in SEGS:
                sel = [i for i, t in enumerate(tok_lab) if t == seg and i > 0]
                row[f"{lab}__{seg}__sum"] = float(lp[sel].sum())
                row[f"{lab}__{seg}__n"] = len(sel)
        rows.append(row)
        print(idx, flush=True) if idx % 20 == 0 else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": {"base": args.base, "adapters": adapters, "data": str(DATA.relative_to(REPO)),
                                                   "n": len(rows), "n_val": len(val), "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                          "rows": rows}, indent=1))
    print("done", args.out)


if __name__ == "__main__":
    main()
