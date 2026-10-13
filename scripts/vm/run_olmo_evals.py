#!/usr/bin/env python3
"""OLMo-3-7B-Think Base and SFT on Haskins (standalone) and ReasonIF, using the same notebooks and decoding settings as the
Qwen3-14B runs. Only the model id, the adapter path, the output location and the model-family checks are changed.
Results are copied to ~/Benign_CoT_Control/results/olmo3_7b_think/ (never overwriting)."""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("rse", str(Path(__file__).resolve().parent / "run_seed_evals.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

WORK = Path.home() / "olmo_runs"
R.WORK, R.LOG = WORK, WORK / "eval_progress.log"
R.DEST = R.REPO / "results" / "olmo3_7b_think"
OLMO = "allenai/Olmo-3-7B-Think"
LORA = str(WORK / "olmo3-7b-think-gpt52-high-reasoning-original")

HASKINS_FAMILY = {
    'MODEL_ID = "Qwen/Qwen3-14B"': f'MODEL_ID = "{OLMO}"',
    '    "Qwen/Qwen3-14B": "qwen3",': f'    "Qwen/Qwen3-14B": "qwen3",\n    "{OLMO}": "olmo3",',
    'if MODEL_FAMILY == "qwen3":\n    assert "Qwen3ForCausalLM" in architectures, architectures\nelse:':
        'if MODEL_FAMILY == "qwen3":\n    assert "Qwen3ForCausalLM" in architectures, architectures\n'
        'elif MODEL_FAMILY == "olmo3":\n    assert "Olmo3ForCausalLM" in architectures, architectures\nelse:',
}


def haskins(tag, lora_value, out_root):
    reps = dict(HASKINS_FAMILY)
    reps['LORA_PATH = "kyleliu789/qwen3-14b-gpt52-high-reasoning-original"'] = f'LORA_PATH = "{lora_value}"'
    reps['OUTPUT_ROOT = Path("haskins_vllm_runs")'] = f'OUTPUT_ROOT = Path("{out_root}")'
    src = R.NB / "Haskins_500_vLLM_Evaluation.ipynb"
    nb_path, kernel = R.patched_copy(src, f"OLMO3_{tag}.ipynb", reps)
    rc = R.execute(nb_path, kernel, tag)
    runs = sorted((R.NB / out_root).glob("*/COMPLETE.json"))
    if rc != 0 or len(runs) != 1:
        R.log(f"FAILED {tag}: rc={rc}, complete runs={runs}")
        return False
    R.copy_run(runs[0].parent, "haskins_500")
    return True


def reasonif(tag, sft, lora=LORA, dest_subdir="reasonif_300"):
    reps = {'FULL_MODEL_ID = "Jackrong/gpt-oss-120b-Distill-Qwen3-4B-Thinking"': f'FULL_MODEL_ID = "{OLMO}"',
            'BASE_MODEL_ID = "Qwen/Qwen3-14B"': f'BASE_MODEL_ID = "{OLMO}"'}
    if sft:
        reps.update({
            "LORA_MODEL_ID = 'kyleliu789/qwen3-14b-qwen3-gpt-52-long-high-reasoning-original'":
                f"LORA_MODEL_ID = 'local/{Path(lora).name}'",
            "target_info = api.model_info(MODEL_ID, token=HF_TOKEN)\nresolved_revision = MODEL_REVISION or target_info.sha":
                "resolved_revision = 'local'",
            "lora_path = snapshot_download(repo_id=LORA_MODEL_ID, revision=resolved_revision, token=HF_TOKEN)": f"lora_path = '{lora}'",
        })
    else:
        reps['EVAL_MODE = "lora"  # "full" or "lora"'] = 'EVAL_MODE = "full"  # "full" or "lora"'
    out_base = WORK / f"reasonif_{tag}"
    reps["OUTPUT_BASE = None"] = f'OUTPUT_BASE = "{out_base}"'
    nb_path, kernel = R.patched_copy(R.RIF / "reasonif_qwen3_14b_hf_final_paper_eval.ipynb", f"OLMO3_{tag}.ipynb", reps)
    rc = R.execute(nb_path, kernel, tag)
    scored = sorted(out_base.glob("*/*/scored_responses.jsonl"))
    if rc != 0 or len(scored) != 1:
        R.log(f"FAILED {tag}: rc={rc}, scored files={scored}")
        return False
    R.copy_run(scored[0].parent, dest_subdir)
    return True


def main(which):
    R.DEST.mkdir(parents=True, exist_ok=True)
    jobs = {"haskins_base": lambda: haskins("haskins_base", "", "haskins_vllm_runs_olmo3_base"),
            "reasonif_base": lambda: reasonif("reasonif_base", sft=False),
            "haskins_sft": lambda: haskins("haskins_sft", LORA, "haskins_vllm_runs_olmo3_sft"),
            "reasonif_sft": lambda: reasonif("reasonif_sft", sft=True),
            "haskins_sft_lr2e-4": lambda: haskins("haskins_sft_lr2e-4", str(WORK / "olmo3-7b-think-gpt52-high-reasoning-original-lr2e-4"), "haskins_vllm_runs_olmo3_sft_lr2e-4"),
            "reasonif_sft_lr2e-4": lambda: reasonif("reasonif_sft_lr2e-4", sft=True,
                                                     lora=str(WORK / "olmo3-7b-think-gpt52-high-reasoning-original-lr2e-4"),
                                                     dest_subdir="reasonif_300_sft_lr2e-4")}
    names = list(jobs) if which == "all" else which.split(",")
    ok = [jobs[n]() for n in names]
    R.log(f"ALL DONE ({which}) ok={all(ok)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
