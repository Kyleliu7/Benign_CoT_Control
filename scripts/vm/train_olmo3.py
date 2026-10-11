"""LoRA SFT of allenai/Olmo-3-7B-Think on the same 212-example dataset with the same hyperparameters as the Qwen3-14B run.

LlamaFactory has no OLMo-3 template, so this registers `olmo3_think`, which reproduces the model's official chat template
(chat_template.jinja at revision d97e442d): system turns get the " You do not currently have access to any functions.
<functions></functions>" suffix, the default system prompt is used when none is given (the training data have none), and the
final assistant turn ends with <|endoftext|>. The training outputs already start with "<think>", matching the official
generation prompt "<|im_start|>assistant\n<think>".

Usage: python train_olmo3.py check            # compare LlamaFactory encoding with the official template, no training
       python train_olmo3.py train <out_dir> [learning_rate]
"""
import sys
from pathlib import Path

import yaml
from llamafactory.data.formatter import StringFormatter
from llamafactory.data.template import register_template

MODEL = "allenai/Olmo-3-7B-Think"
REVISION = "d97e442d7cc678210054dbcc9b440894d62c89a4"
OLMO_DEFAULT_SYSTEM = ("You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, "
                       "and your model weights are available at https://huggingface.co/allenai.")

register_template(
    name="olmo3_think",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}} You do not currently have access to any functions. "
                                         "<functions></functions><|im_end|>\n"]),
    default_system=OLMO_DEFAULT_SYSTEM,
    stop_words=["<|im_end|>"],
)


def check():
    import json
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tmpl = TEMPLATES["olmo3_think"]
    data = json.load(open("data/training/gpt52_high_reasoning_original.json"))
    mismatches = 0
    for ex in data[:20]:
        user = ex["instruction"] + (("\n" + ex["input"]) if ex.get("input") else "")
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": ex["output"]}]
        p_ids, r_ids = tmpl.encode_oneturn(tok, msgs)
        ref = tok.apply_chat_template(msgs, tokenize=False)
        ref_ids = tok.encode(ref, add_special_tokens=False)
        if p_ids + r_ids != ref_ids:
            mismatches += 1
            print("MISMATCH\nLF :", repr(tok.decode(p_ids + r_ids)[:400]), "\nREF:", repr(ref[:400]))
            print("first differing index:", next(i for i, (a, b) in enumerate(zip(p_ids + r_ids, ref_ids)) if a != b) if any(
                a != b for a, b in zip(p_ids + r_ids, ref_ids)) else f"length {len(p_ids + r_ids)} vs {len(ref_ids)}")
    gen = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    print("official generation prompt ends with:", repr(gen[-40:]))
    print("LlamaFactory prompt ends with:      ", repr(tok.decode(tmpl.encode_oneturn(tok, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "<think>x"}])[0])[-40:]))
    print(f"{mismatches} mismatches in 20 examples")


def train(out_dir, lr=None):
    cfg = yaml.safe_load(open("configs/qwen3_14b_sft_lora-gpt-52-high-reasoning-original.yaml"))
    cfg.update(model_name_or_path=MODEL, model_revision=REVISION, template="olmo3_think", output_dir=out_dir)
    if lr is not None:
        cfg["learning_rate"] = float(lr)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "train_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    from llamafactory.train.tuner import run_exp
    run_exp(cfg)


if __name__ == "__main__":
    check() if sys.argv[1] == "check" else train(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
