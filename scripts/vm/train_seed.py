"""Retrain a LoRA adapter with a new training seed while keeping the seed-42 validation split.

LlamaFactory uses training_args.seed both for the train/val split and for training randomness
(LoRA init, data order, dropout). This wrapper passes seed=42 to the split only, so the 22 held-out
examples are the same as in the original run and only the training randomness changes.
Usage: python train_seed.py <config.yaml> <seed> <output_dir>
"""
import json
import sys
from pathlib import Path

import yaml
import llamafactory.data.loader as loader

SPLIT_SEED = 42
cfg_path, seed, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
Path(out_dir).mkdir(parents=True, exist_ok=True)

_orig_split = loader.split_dataset


def split_with_fixed_seed(dataset, eval_dataset, data_args, seed):
    train_dict, eval_dict = _orig_split(dataset, eval_dataset, data_args, seed=SPLIT_SEED)
    ev = eval_dict.get("validation")
    if ev is not None:
        # record the held-out examples (token ids hashed) so the split can be compared across runs
        import hashlib
        hashes = sorted(hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:16] for r in ev)
        Path(out_dir, "val_split_hashes.json").write_text(json.dumps({"split_seed": SPLIT_SEED, "n_val": len(ev), "n_train": len(train_dict["train"]), "hashes": hashes}, indent=1))
    return train_dict, eval_dict


loader.split_dataset = split_with_fixed_seed

cfg = yaml.safe_load(open(cfg_path))
cfg["seed"] = seed
cfg["output_dir"] = out_dir
Path(out_dir, "train_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

from llamafactory.train.tuner import run_exp  # noqa: E402

run_exp(cfg)
