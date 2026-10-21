#!/usr/bin/env python3
"""CoT-Control (Haskins 500, standalone) for a Phi-4-reasoning LoRA adapter trained here, with the notebook and settings
of the reported Phi SFT run (Haskins_500_Phi4_Evaluation.ipynb). Only the adapter path and output location change; the
finished run is copied to <DEST>/haskins_500/phi4_reasoning/ (never overwriting).

  VARIANT=s1k_claude37 python run_phi_variant_evals.py      (LORA default ~/variant_runs/phi4-reasoning-<VARIANT>,
                                                            DEST default ~/Benign_CoT_Control/results/data_variants/phi4_<VARIANT>)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TRAIN_SEED", "42")
import run_seed_evals as rse  # noqa: E402

VARIANT = os.environ["VARIANT"]
rse.WORK = Path.home() / "variant_runs"
rse.LOG = rse.WORK / "eval_progress.log"
rse.DEST = Path(os.environ.get("DEST", str(rse.REPO / "results" / "data_variants" / f"phi4_{VARIANT}")))
rse.SEED = f"phi_{VARIANT}"
LORA = os.environ.get("LORA", str(rse.WORK / f"phi4-reasoning-{VARIANT}"))

if __name__ == "__main__":
    assert Path(LORA).is_dir(), LORA
    rse.DEST.mkdir(parents=True, exist_ok=True)
    ok = rse.haskins_job(f"phi_haskins_standalone_{VARIANT}", "Haskins_500_Phi4_Evaluation.ipynb",
                         "LORA_PATH = 'kyleliu789/phi4-reasoning-14b-gpt52-high-reasoning-original'",
                         f"LORA_PATH = '{LORA}'", f"haskins_phi4_runs_{VARIANT}_standalone", "haskins_500/phi4_reasoning")
    rse.log(f"ALL DONE (phi {VARIANT}) ok={ok}")
    sys.exit(0 if ok else 1)
