#!/usr/bin/env python3
"""Evaluate the seed-2026 adapters with the same notebooks that produced the reported seed-42 results.

Each job copies the original notebook next to itself (so relative paths resolve), changes only the adapter
path and the output location, executes it with nbconvert, and copies the finished run into
~/Benign_CoT_Control/results/seed2026/. Existing result directories are never touched.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = HOME / "Benign_CoT_Control"
NB = REPO / "notebooks"
RIF = HOME / "uncontaminated_workspace" / "reasonif"
JUPYTER = HOME / "jupyter-env" / "bin" / "jupyter"
SEED = int(os.environ.get("TRAIN_SEED", "2026"))
DEST = REPO / "results" / "training_seeds" / f"seed{SEED}"
WORK = HOME / "seed_runs"
QWEN_LORA = str(WORK / f"qwen3-14b-seed{SEED}")
PHI_LORA = str(WORK / f"phi4-reasoning-seed{SEED}")
LOG = WORK / "eval_progress.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def patched_copy(src, dst_name, replacements):
    nb = json.loads(src.read_text(encoding="utf-8"))
    remaining = dict(replacements)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
                remaining.pop(old, None)
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    assert not remaining, f"{src.name}: patterns not found: {list(remaining)}"
    dst = src.parent / dst_name
    dst.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    kernel = nb.get("metadata", {}).get("kernelspec", {}).get("name") or "haskins-vllm"
    return dst, kernel


def execute(nb_path, kernel, tag):
    out = WORK / "executed" / f"{tag}.ipynb"
    out.parent.mkdir(exist_ok=True)
    cmd = [str(JUPYTER), "nbconvert", "--to", "notebook", "--execute",
           f"--ExecutePreprocessor.kernel_name={kernel}", "--ExecutePreprocessor.timeout=-1",
           str(nb_path), "--output", str(out)]
    log(f"START {tag} ({nb_path.name}, kernel {kernel})")
    t = time.time()
    with (WORK / f"{tag}.log").open("w") as f:
        rc = subprocess.run(cmd, cwd=str(nb_path.parent), stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"END {tag} rc={rc} in {(time.time() - t) / 60:.1f} min")
    return rc


def copy_run(src_dir, dest_subdir):
    dest = DEST / dest_subdir / src_dir.name
    assert not dest.exists(), f"refusing to overwrite {dest}"
    shutil.copytree(src_dir, dest)
    log(f"copied {src_dir} -> {dest}")


def haskins_job(tag, nb_name, lora_old, lora_new, out_root, dest_subdir):
    src = NB / nb_name
    root_old = 'OUTPUT_ROOT = Path("haskins_vllm_runs")' if "Phi4" not in nb_name else None
    reps = {lora_old: lora_new}
    text = src.read_text(encoding="utf-8")
    for cand in ['OUTPUT_ROOT = Path("haskins_vllm_runs")', "OUTPUT_ROOT = Path('haskins_phi4_runs')",
                 'OUTPUT_ROOT = Path("haskins_phi4_runs")']:
        if cand.replace('"', '\\"') in text or cand in text:
            reps[cand] = f'OUTPUT_ROOT = Path("{out_root}")'
            break
    else:
        raise AssertionError(f"no OUTPUT_ROOT line found in {nb_name} ({root_old})")
    nb_path, kernel = patched_copy(src, f"SEED{SEED}_{tag}.ipynb", reps)
    rc = execute(nb_path, kernel, tag)
    if rc != 0 and not os.environ.get("VLLM_USE_AOT_COMPILE"):
        log(f"RETRY {tag} with VLLM_USE_AOT_COMPILE=0")
        os.environ["VLLM_USE_AOT_COMPILE"] = "0"
        rc = execute(nb_path, kernel, tag + "_retry")
        del os.environ["VLLM_USE_AOT_COMPILE"]
    runs = sorted((NB / out_root).glob("*/COMPLETE.json"))
    if rc != 0 or len(runs) != 1:
        log(f"FAILED {tag}: rc={rc}, complete runs={runs}")
        return False
    copy_run(runs[0].parent, dest_subdir)
    return True


def reasonif_job(tag, nb_rel, reps, out_base, dest_subdir):
    src = RIF / nb_rel
    reps = dict(reps)
    reps["OUTPUT_BASE = None"] = f'OUTPUT_BASE = "{out_base}"'
    nb_path, kernel = patched_copy(src, f"SEED{SEED}_{tag}.ipynb", reps)
    rc = execute(nb_path, kernel, tag)
    scored = sorted(Path(out_base).glob("*/*/scored_responses.jsonl"))
    if rc != 0 or len(scored) != 1:
        log(f"FAILED {tag}: rc={rc}, scored files={scored}")
        return False
    copy_run(scored[0].parent, dest_subdir)
    return True


QWEN_ONLY_STANDALONE = [False]


def main(which):
    DEST.mkdir(parents=True, exist_ok=True)
    jobs = []
    if which == "qwen_standalone":
        which = "qwen"
        QWEN_ONLY_STANDALONE[0] = True
    if which in ("qwen", "all"):
        jobs += [
            lambda: haskins_job("qwen_haskins_standalone_sft", "Haskins_500_vLLM_Evaluation.ipynb",
                                'LORA_PATH = "kyleliu789/qwen3-14b-gpt52-high-reasoning-original"',
                                f'LORA_PATH = "{QWEN_LORA}"', f"haskins_vllm_runs_seed{SEED}_standalone",
                                "haskins_500/qwen3_14b"),
            lambda: haskins_job("qwen_haskins_sft_to_base", "Haskins_500_vLLM_2x2_SFT_Donor_to_Base_Evaluation.ipynb",
                                'LORA_PATH = "kyleliu789/qwen3-14b-gpt52-high-reasoning-original"',
                                f'LORA_PATH = "{QWEN_LORA}"', f"haskins_vllm_runs_seed{SEED}_sft_to_base",
                                "haskins_500/qwen3_14b"),
            lambda: reasonif_job("qwen_reasonif_sft", "reasonif_qwen3_14b_hf_final_paper_eval.ipynb", {
                "LORA_MODEL_ID = 'kyleliu789/qwen3-14b-qwen3-gpt-52-long-high-reasoning-original'":
                    f"LORA_MODEL_ID = 'local/qwen3-14b-gpt52-high-reasoning-seed{SEED}'",
                "target_info = api.model_info(MODEL_ID, token=HF_TOKEN)\nresolved_revision = MODEL_REVISION or target_info.sha":
                    f"resolved_revision = 'local-seed{SEED}'",
                "lora_path = snapshot_download(repo_id=LORA_MODEL_ID, revision=resolved_revision, token=HF_TOKEN)":
                    f"lora_path = '{QWEN_LORA}'",
            }, str(WORK / f"reasonif_qwen_seed{SEED}"), "reasonif_300/qwen3_14b"),
        ]
    if which in ("phi", "all"):
        jobs += [
            lambda: haskins_job("phi_haskins_standalone_sft", "Haskins_500_Phi4_Evaluation.ipynb",
                                "LORA_PATH = 'kyleliu789/phi4-reasoning-14b-gpt52-high-reasoning-original'",
                                f"LORA_PATH = '{PHI_LORA}'", f"haskins_phi4_runs_seed{SEED}_standalone",
                                "haskins_500/phi4_reasoning"),
            lambda: haskins_job("phi_haskins_sft_to_base", "Haskins_500_Phi4_2x2_SFT_Donor_to_Base_Evaluation.ipynb",
                                'LORA_PATH = "kyleliu789/phi4-reasoning-14b-gpt52-high-reasoning-original"',
                                f'LORA_PATH = "{PHI_LORA}"', f"haskins_phi4_runs_seed{SEED}_sft_to_base",
                                "haskins_500/phi4_reasoning"),
            lambda: reasonif_job("phi_reasonif_sft", "phi/ReasonIF_Phi4_SFT_Evaluation.ipynb", {
                'LORA_MODEL_ID = "kyleliu789/phi4-reasoning-14b-gpt52-high-reasoning-original"':
                    f'LORA_MODEL_ID = "local/phi4-reasoning-14b-gpt52-high-reasoning-seed{SEED}"\nLOCAL_LORA_PATH = "{PHI_LORA}"',
            }, str(WORK / f"reasonif_phi_seed{SEED}"), "reasonif_300/phi4_reasoning"),
        ]
    if QWEN_ONLY_STANDALONE[0]:
        jobs = [jobs[0], jobs[2]]
    ok = all([job() for job in jobs])
    log(f"ALL DONE ({which}) ok={ok}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
