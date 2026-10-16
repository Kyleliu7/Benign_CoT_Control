#!/usr/bin/env python3
"""Evaluate a Qwen3-14B LoRA adapter (a data variant trained here, or an existing adapter on the Hub) with the same
notebooks and settings as the reported SFT adapter: CoT-Control (Haskins 500, standalone) and ReasonIF 300.

Same mechanics as run_seed_evals.py: each notebook is copied, only the adapter path and output location are
changed, it is executed with nbconvert, and the finished run is copied to <DEST>/haskins_500/qwen3_14b/ and
<DEST>/reasonif_300/qwen3_14b/. Existing result directories are never touched.

Environment:
  VARIANT  name used for output folders (required)
  LORA     adapter: a local directory or a Hub repo id (default ~/variant_runs/qwen3-14b-<VARIANT>)
  DEST     destination root (default ~/Benign_CoT_Control/results/data_variants/<VARIANT>)
  JOBS     comma-separated subset of "haskins,reasonif" (default both; ReasonIF needs a local LORA directory)

  VARIANT=no_headers python run_variant_evals.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TRAIN_SEED", "42")
import run_seed_evals as rse  # noqa: E402

VARIANT = os.environ["VARIANT"]
rse.WORK = Path.home() / "variant_runs"
rse.WORK.mkdir(exist_ok=True)
rse.LOG = rse.WORK / "eval_progress.log"
rse.DEST = Path(os.environ.get("DEST", str(rse.REPO / "results" / "data_variants" / VARIANT)))
rse.SEED = VARIANT  # only used in notebook copy names
LORA = os.environ.get("LORA", str(rse.WORK / f"qwen3-14b-{VARIANT}"))
JOBS = os.environ.get("JOBS", "haskins,reasonif").split(",")


def main():
    rse.DEST.mkdir(parents=True, exist_ok=True)
    ok = True
    if "haskins" in JOBS:
        ok = rse.haskins_job(f"qwen_haskins_standalone_{VARIANT}", "Haskins_500_vLLM_Evaluation.ipynb",
                             'LORA_PATH = "kyleliu789/qwen3-14b-gpt52-high-reasoning-original"',
                             f'LORA_PATH = "{LORA}"', f"haskins_vllm_runs_{VARIANT}_standalone", "haskins_500/qwen3_14b") and ok
    if "reasonif" in JOBS:
        assert Path(LORA).is_dir(), "ReasonIF job needs a local adapter directory"
        ok = rse.reasonif_job(f"qwen_reasonif_{VARIANT}", "reasonif_qwen3_14b_hf_final_paper_eval.ipynb", {
            "LORA_MODEL_ID = 'kyleliu789/qwen3-14b-qwen3-gpt-52-long-high-reasoning-original'":
                f"LORA_MODEL_ID = 'local/qwen3-14b-{VARIANT}'",
            "target_info = api.model_info(MODEL_ID, token=HF_TOKEN)\nresolved_revision = MODEL_REVISION or target_info.sha":
                f"resolved_revision = 'local-{VARIANT}'",
            "lora_path = snapshot_download(repo_id=LORA_MODEL_ID, revision=resolved_revision, token=HF_TOKEN)":
                f"lora_path = '{LORA}'",
        }, str(rse.WORK / f"reasonif_qwen_{VARIANT}"), "reasonif_300/qwen3_14b") and ok
    rse.log(f"ALL DONE ({VARIANT}) ok={ok}")


if __name__ == "__main__":
    main()
