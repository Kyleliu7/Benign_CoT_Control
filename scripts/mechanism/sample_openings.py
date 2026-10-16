"""Sample the opening of the reasoning (first 150 new tokens) on CoT-Control prompts, for a quick look at how a model
starts its chain of thought (e.g. whether it writes a bold heading and whether the heading names the constraint).

Items: questions 0-4 of each of the ten tasks (50 items), the exact evaluation prompts (haskins_prompts.py), sampled with
the CoT-Control settings (temperature 0.6, top-p 1.0), one sample per item, torch seed 42 + item index.

  python sample_openings.py --base Qwen/Qwen3.5-9B [--adapter path] --out <file.jsonl>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from haskins_prompts import items, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--strip-template-newline", action="store_true",
                    help="drop the newline of a template-final '<think>\\n' (Qwen3.5), matching the training format '<think>**...'")
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from haskins_prompts import load_causal_lm
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = load_causal_lm(args.base, torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    sel = [it for it in items() if it["q"] < args.per_task]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, it in enumerate(sel):
            prompt = render(tok, it["messages"])
            if args.strip_template_newline and prompt.endswith("<think>\n"):
                prompt = prompt[:-1]
            x = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            torch.manual_seed(42 + i)
            with torch.no_grad():
                y = model.generate(**x, do_sample=True, temperature=0.6, top_p=1.0, top_k=0,
                                   max_new_tokens=args.max_new_tokens)
            text = tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=False)
            f.write(json.dumps({"task": it["task"], "q": it["q"], "base": args.base, "adapter": args.adapter,
                                "prompt_ends_with": prompt[-40:], "opening": text}, ensure_ascii=False) + "\n")
    print("done", args.out)


if __name__ == "__main__":
    main()
