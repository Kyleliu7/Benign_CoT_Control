"""Probability that the first token of the reasoning starts a bold heading ("**"), base vs adapters, for any model.

Input: the 500 CoT-Control evaluation prompts (haskins_prompts.py) rendered with the model's own chat template and
generation prompt, followed by the reasoning opener if the template does not already end with it ("<think>\n" for
Qwen-style templates). Output: for each item, the total probability of vocabulary tokens whose text starts with "**"
(the heading opener) and the top-1 token, for the base model (adapters off) and each adapter.

  python first_token_probe.py --base allenai/Olmo-3-7B-Think --adapter lr1e-4=<path> ... --out <file.json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from haskins_prompts import items, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", action="append", default=[], help="name=path")
    ap.add_argument("--opener", default="<think>\n", help="appended unless the rendered prompt already ends with <think>")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    adapters = dict(a.split("=", 1) for a in args.adapter)
    for i, (n, p) in enumerate(adapters.items()):
        model = PeftModel.from_pretrained(model, p, adapter_name=n) if i == 0 else model
        if i > 0:
            model.load_adapter(p, adapter_name=n)
    model.eval()
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    star_ids = [i for i, t in enumerate(vocab) if t is not None and tok.convert_tokens_to_string([t]).lstrip().startswith("**")]
    rows = []
    for it in items():
        prompt = render(tok, it["messages"])
        text = prompt if prompt.rstrip().endswith("<think>") else prompt + args.opener
        x = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        row = {"task": it["task"], "q": it["q"], "prompt_tail": text[-30:]}
        for lab in ["base"] + list(adapters):
            if lab == "base":
                ctx = model.disable_adapter() if adapters else torch.no_grad()
            else:
                model.set_adapter(lab)
                ctx = torch.no_grad()
            with ctx, torch.no_grad():
                p = torch.softmax(model(**x).logits[0, -1].float(), -1)
            row[f"{lab}__p_star"] = float(p[star_ids].sum())
            row[f"{lab}__top1"] = tok.decode([int(p.argmax())])
        rows.append(row)
    Path(args.out).write_text(json.dumps({"meta": {"base": args.base, "adapters": adapters, "n_star_tokens": len(star_ids),
                                                   "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "rows": rows}, indent=1))
    print("done", args.out)


if __name__ == "__main__":
    main()
