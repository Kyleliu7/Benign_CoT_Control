"""Attention at the start of the reasoning, Qwen3-14B base vs LoRA adapters, on identical inputs.

Part 1 (measurement). For each of the 500 CoT-Control items the input is the exact evaluation prompt plus "<think>\n",
the point at which the model writes its first reasoning token. With adapters off (base) and on (each adapter), we
record, for every layer and head, the attention weight that the last position puts on (a) the constraint instruction,
(b) the question and (c) everything else, plus the probability that the next token is "**" (a bold heading).

Part 2 (attention-pattern patching, after the method in abhayesian's LessWrong shortform on steering CoT
controllability). The opening O is the first 10 reasoning tokens that the SFT model actually wrote for that item
(results/haskins_500/qwen3_14b/haskins_qwen3_14b_vllm_calibrated_sft_31pct.jsonl). We teacher-force prompt +
"<think>\n" + O and compute log p(O) under: base; SFT; base with the SFT model's attention probabilities substituted
in a set of layers (values, MLPs and everything else stay base); and SFT with the base model's attention substituted.
"Gap recovered" = sum(patched - base) / sum(SFT - base). Sanity check: substituting a model's own attention must
leave log p unchanged.

  python attention_reasoning_start.py --adapter sft42=<path> [--adapter name=path ...] --out <dir> [--patch-adapter sft42]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from haskins_prompts import REPO, items, render  # noqa: E402

BASE = "Qwen/Qwen3-14B"
SFT_RECORDS = REPO / "results/haskins_500/qwen3_14b/haskins_qwen3_14b_vllm_calibrated_sft_31pct.jsonl"
STATE = {"mode": None, "store": {}, "layers": set()}


def install_hook():
    import transformers.models.qwen3.modeling_qwen3 as mq
    orig = mq.eager_attention_forward

    def hooked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, w = orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
        li = module.layer_idx
        if STATE["mode"] == "record":
            STATE["store"][li] = w.detach()
        elif STATE["mode"] == "patch" and li in STATE["layers"]:
            w = STATE["store"][li].to(w.dtype)
            v = mq.repeat_kv(value, module.num_key_value_groups)
            out = torch.matmul(w, v).transpose(1, 2).contiguous()
        return out, w

    mq.eager_attention_forward = hooked


def spans(tok, text, parts):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    masks = {}
    for name, sub in parts.items():
        a = text.index(sub)
        b = a + len(sub)
        masks[name] = np.array([s < b and e > a for s, e in offs])
    return ids, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[], help="name=path")
    ap.add_argument("--patch-adapter", default="sft42")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    install_hook()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    adapters = dict(a.split("=", 1) for a in args.adapter)
    first = True
    for name, path in adapters.items():
        if first:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
            first = False
        else:
            model.load_adapter(path, adapter_name=name)
    model.eval()
    star = tok.encode("**", add_special_tokens=False)
    assert len(star) == 1, star
    star = star[0]

    its = items()
    if args.limit:
        its = its[:: max(1, len(its) // args.limit)][: args.limit]
    meta = {"base": BASE, "adapters": adapters, "n_items": len(its), "transformers_attn": "eager", "star_token": star,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---------------- Part 1: attention at the first reasoning position ----------------
    labels = ["base"] + list(adapters)
    res = {k: {"constraint": [], "question": [], "p_star": [], "top1": []} for k in labels}
    rows = []
    for it in its:
        text = render(tok, it["messages"]) + "<think>\n"
        ids, m = spans(tok, text, {"constraint": it["instruction"], "question": it["question"]})
        x = torch.tensor([ids], device="cuda")
        rows.append({"task": it["task"], "q": it["q"], "n_tokens": len(ids),
                     "n_constraint": int(m["constraint"].sum()), "n_question": int(m["question"].sum())})
        for lab in labels:
            ctx = model.disable_adapter() if lab == "base" else torch.no_grad()
            if lab != "base":
                model.set_adapter(lab)
            with ctx, torch.no_grad():
                o = model(x, output_attentions=True)
            att = torch.stack([a[0, :, -1, :].float() for a in o.attentions]).cpu().numpy()  # [L, H, T]
            res[lab]["constraint"].append(att[:, :, m["constraint"]].sum(-1))
            res[lab]["question"].append(att[:, :, m["question"]].sum(-1))
            p = torch.softmax(o.logits[0, -1].float(), -1)
            res[lab]["p_star"].append(float(p[star]))
            res[lab]["top1"].append(tok.decode([int(p.argmax())]))
    np.savez_compressed(out / "attention_first_position.npz",
                        **{f"{lab}__{k}": np.array(v) for lab in labels for k, v in res[lab].items() if k != "top1"})
    (out / "attention_first_position_items.json").write_text(json.dumps(
        {"meta": meta, "items": rows, "top1": {lab: res[lab]["top1"] for lab in labels}}, indent=1))
    print("part 1 done", flush=True)

    # ---------------- Part 2: attention-pattern patching on the SFT model's own opening ----------------
    sft = {}
    for line in SFT_RECORDS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            sft[(r["task"], int(r["prompt_idx"]))] = r["reasoning"] or ""
    pa = args.patch_adapter
    model.set_adapter(pa)
    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else model.base_model.config.num_hidden_layers
    t = n_layers // 3
    sets = {"all": range(n_layers), "early": range(0, t), "middle": range(t, 2 * t), "late": range(2 * t, n_layers),
            "last6": range(n_layers - 6, n_layers)}

    def logp(x, n_open):
        o = model(x)
        lp = torch.log_softmax(o.logits[0, -n_open - 1:-1].float(), -1)
        return float(lp.gather(1, x[0, -n_open:, None]).sum())

    prow = []
    for it in its:
        reasoning = sft.get((it["task"], it["q"]), "")
        open_ids = tok.encode(reasoning.lstrip("\n"), add_special_tokens=False)[:10]
        if len(open_ids) < 10:
            continue
        pre = tok.encode(render(tok, it["messages"]) + "<think>\n", add_special_tokens=False)
        x = torch.tensor([pre + open_ids], device="cuda")
        r = {"task": it["task"], "q": it["q"], "opening": tok.decode(open_ids)}
        with torch.no_grad():
            STATE["mode"] = "record"
            with model.disable_adapter():
                r["base"] = logp(x, 10)
            base_store = dict(STATE["store"])
            STATE["store"] = {}
            r["sft"] = logp(x, 10)
            sft_store = dict(STATE["store"])
            # sanity: own attention substituted
            STATE["mode"], STATE["layers"], STATE["store"] = "patch", set(range(n_layers)), base_store
            with model.disable_adapter():
                r["base_self_patch"] = logp(x, 10)
            for sname, ls in sets.items():
                STATE["layers"], STATE["store"] = set(ls), sft_store
                with model.disable_adapter():
                    r[f"base_with_sft_attn_{sname}"] = logp(x, 10)
                STATE["store"] = base_store
                r[f"sft_with_base_attn_{sname}"] = logp(x, 10)
            STATE["mode"], STATE["store"] = None, {}
        prow.append(r)
    (out / "attention_patching.json").write_text(json.dumps({"meta": {**meta, "patch_adapter": pa, "layer_sets":
                                                  {k: [min(v), max(v)] for k, v in sets.items()}}, "rows": prow}, indent=1))
    print("part 2 done", len(prow), "items", flush=True)


if __name__ == "__main__":
    main()
