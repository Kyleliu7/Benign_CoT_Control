# VM index (home directory of kyleliu789)

Two VMs share this layout: `instance-20260831-015150` (us-central1-c, original disk) and `gpu-vm-useast4c`
(us-east4-c, created 2026-10-16 from a snapshot of the original disk). Runs after 2026-10-16 01:00 UTC exist only on the
VM that ran them; every result was also copied into the GitHub repository (Kyleliu7/Benign_CoT_Control, branch
`claude/quirky-darwin-tsttu0`) and into `vm_evidence/` archives on Kyle's laptop. Nothing here was moved or deleted;
old notebooks reference these paths.

| Path | What it is | Code (repository) |
|---|---|---|
| `Benign_CoT_Control/` | Evaluation checkout used by every notebook run (an older commit plus generated notebooks). `notebooks/` holds the run folders the notebooks write; `results/` the copies made by the run scripts. | notebooks/, scripts/vm/ |
| `uncontaminated_workspace/reasonif/` | ReasonIF evaluation notebooks (Qwen, Phi, OLMo). | notebooks referenced in scripts/vm/run_*_evals.py |
| `workspace/LlamaFactory/` | LlamaFactory install, the original training configs and data (`data/*.json`, including the other-teacher datasets), and the Qwen3.5 adapters. | configs/, data/training/ |
| `seed_runs/` | Qwen3-14B training seeds 1, 2 (and the stopped 2026 run): adapters, logs. | scripts/vm/train_seed.py, run_seed_evals.py |
| `olmo_runs/` | OLMo-3-7B-Think adapters (lr 1e-4, 2e-4) and their evaluation logs. | scripts/vm/train_olmo3.py, run_olmo_evals.py |
| `variant_runs/` | Header ablations (`qwen3-14b-no_headers` on us-east4-c, `qwen3-14b-neutral_header` and `qwen3-14b-header_transplant` on us-central1-c), their logs, and copies of the run scripts. `attempt1_offline_error/` keeps the logs of the first no-header evaluation attempt, which failed because offline mode was left on. | scripts/vm/run_variant_chain.sh, run_variant_evals.py |
| `repo_snapshots/<commit>/` | Exact repository snapshot (git archive) that the 2026-10-16 mechanism session ran from; its `results/mechanism/` holds that session's analysis outputs. | scripts/mechanism/, scripts/vm/mechanism_chain.sh |
| `mechanism_runs/` | Logs and status of the 2026-10-16 session (`status`, one log per step, `paths.txt` with the adapter paths used). | scripts/vm/mechanism_chain.sh, probe_first_token.sh |
| `vm_untracked_backup_20261010/` | Backup of run folders that were untracked on 2026-10-10 (identical to the committed ones). | |
| `.cache/huggingface/hub/` | Base models and published adapters (pinned revisions are listed in the repository README). | |
| `llamafactory-env/`, `jupyter-env/`, `venvs/` | Python environments: training and analysis (`llamafactory-env`), notebook execution (`jupyter-env`), vLLM kernels (`venvs/haskins-vllm`). | requirements*.txt |
| top-level `*.py`, `*.log` | Earlier one-off checks and inspection scripts from the original project; kept as they were. | |
