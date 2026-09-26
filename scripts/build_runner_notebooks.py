"""Generate the Colab and Kaggle GPU runner notebooks from verified sources.

The notebooks are self-contained: they clone the official MedMNIST experiment
code at the pinned commit, apply the tested compatibility patches, and process
the preregistered experiment queue with end-of-epoch checkpointing so a
free-tier session disconnect cannot silently discard a run. They contain no
results and they never mark a run verified without artifacts.
"""
import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import experiments
import patch_official
import runner_safety

COMMIT = '70b6b3a7ad7afddff1df2a3b735235830fbdb142'
DRIVE_FOLDER = 'medmnist_benchmark'
REPO_URL = 'https://github.com/MedMNIST/experiments'

PRIOR_FAILED = {
    '2d-chestmnist-resnet50-224-seed17-r1': (
        'Three prior local attempts failed before completing one epoch. '
        'Only the retained cloud record is published in this repository.')
}


def embedded_queue_json():
    """Preregistered run queue from the tested experiment plan, chestmnist first."""
    plan = json.loads((ROOT / 'configs/experiment-plan.json').read_text(encoding='utf-8'))
    references = {item['flag']: item for item in plan['datasets']}
    queue = []
    for run in plan['runs']:
        reference = references[run['dataset']]
        item = {'run_id': run['run_id'], 'dataset': run['dataset'], 'seed': run['seed'],
                'dimension': run['dimension'], 'resolution': run['resolution'],
                'batch_size': run['batch_size'],
                'reported_auc': reference['reported_auc'],
                'reported_acc': reference['reported_acc']}
        if run['run_id'] in PRIOR_FAILED:
            item['prior_attempts'] = PRIOR_FAILED[run['run_id']]
        queue.append(item)
    return json.dumps(queue, indent=1)


def _header_md(platform):
    base = [
        '# MedMNIST official baselines runner',
        '',
        'Runs the **official** MedMNIST experiment code (pinned commit) on a free cloud GPU',
        'with end-of-epoch checkpointing, so a session disconnect never silently discards a run.',
        '',
        'Binding rules:',
        '- The baseline phase uses only the official authors\' code plus the documented patches',
        '  written into `LOCAL_PATCHES.md` (seeding + resume; 3D best-checkpoint fix).',
        '- Never fabricate AUC/ACC. A run counts only after its evaluator CSVs and logs',
        '  exist on persistent storage; failed runs stay logged.',
        '- Nothing in this notebook is a result until `status.json` says `verified: true`.',
        '',
        'Parallelism policy (protocol-safe):',
        '- `PARALLEL_RUNS` starts **one independent official run per physical GPU**. Each run',
        '  keeps the official batch size, model and 100 epochs, so results stay comparable',
        '  with the published baselines.',
        '- DataParallel / DDP are deliberately NOT used: they would change effective batch',
        '  size and BatchNorm statistics.',
        '- Every run is claimed atomically (`claim_next`), so two GPUs never share a run_id.',
        '- `DATASETS_THIS_SESSION` / `EXCLUDED_RUN_IDS` scope a session so two separate',
        '  sessions can never write the same run directory.',
        '- Training pauses at an epoch boundary before the session cap; the atomic checkpoint',
        '  makes the next session resume instead of restarting from scratch.',
        '',
    ]
    if platform == 'colab':
        base += [
            'How to use (Colab):',
            '1. Runtime -> Change runtime type -> GPU (T4).',
            '2. Run all cells top to bottom. Data, checkpoints and artifacts persist in the',
            '   Google Drive folder `medmnist_benchmark`.',
            '3. If the session dies, rerun all cells: verified runs are skipped and the',
            '   interrupted run resumes from its last completed epoch.',
            '4. `DIMENSIONS_THIS_SESSION` controls whether the queue processes 2D, 3D, or both.',
        ]
    else:
        base += [
            'How to use (Kaggle):',
            '1. Enable the GPU accelerator (T4 x2 when offered) and internet.',
            '2. Attach exactly one prior state input only when resuming.',
            '3. Start with **Save Version -> Save & Run All**. Kaggle runs the version in a',
            '   fresh session and persists `/kaggle/working` after the notebook exits.',
            '4. For a later version, attach the prior output through one state dataset/input.',
            '   Do not expect Save & Run All to snapshot an already-running interactive session.',
            '5. Rerun this same notebook with the prior state attached; verified runs are skipped',
            '   and `DIMENSIONS_THIS_SESSION` determines whether 2D, 3D, or both are processed.',
        ]
    return '\n'.join(base)


C_CONFIG = '''# ---- Configuration (fixed for the whole official campaign) ----
UPSTREAM_REPO = "https://github.com/MedMNIST/experiments"
UPSTREAM_COMMIT = "COMMIT_PLACEHOLDER"
EPOCHS = 100
SEEDS = [17, 29, 43, 71, 101]
MAX_SESSION_MINUTES = 660      # hard wall-clock budget; below Kaggle's 12h cap so we
                               # can checkpoint, copy artifacts and save the version
FINALIZE_MARGIN_SECONDS = 900  # extra guard between the pause deadline and the cap
MIN_RUN_MINUTES = 45           # do not START a new run with almost no budget left
PARALLEL_RUNS = 2              # ONE official run per physical GPU. Never share a GPU:
                               # sharing would change effective batch size / BatchNorm
                               # statistics and break comparability with the official
                               # baselines. This is NOT DataParallel and NOT DDP.
STOP_ON_FAILURE = True         # stop this worker after a non-verified run
DIMENSION = "2d"               # compatibility/default scope for the standard notebooks
DIMENSIONS_THIS_SESSION = [DIMENSION]
DRIVE_FOLDER = "medmnist_benchmark"
KEEP_BEST_MODEL = True         # best_model.pth per run (~100 MB); set False if storage runs low

# ---- Session scoping (lets two independent sessions work without collisions) ----
# Defaults cover the full official 2D queue. A second, independent session overrides
# DATASETS_THIS_SESSION / EXCLUDED_RUN_IDS so the two never touch the same run_id.
DATASETS_THIS_SESSION = ["chestmnist", "dermamnist", "pathmnist", "octmnist",
                         "pneumoniamnist", "breastmnist", "bloodmnist", "tissuemnist",
                         "organamnist", "organcmnist", "organsmnist", "retinamnist"]
# Seed 17 is already owned by the original active kernel and has a recovered
# epoch-32 checkpoint. This next runner must never duplicate or overwrite it.
EXCLUDED_RUN_IDS = ["2d-chestmnist-resnet50-224-seed17-r1"]
'''

C_PLATFORM_COLAB = '''from pathlib import Path
import os, shutil, time
PLATFORM = "colab"

from google.colab import drive
drive.mount('/content/drive')

BASE = Path('/content/drive/MyDrive') / DRIVE_FOLDER
SESSION_MARKER = Path('/content/.medmnist-session-start')
if not SESSION_MARKER.exists():
    SESSION_MARKER.write_text(str(time.time()))
SESSION_START = float(SESSION_MARKER.read_text())
for sub in ['runs', 'work', 'checkpoints', 'state']:
    (BASE / sub).mkdir(parents=True, exist_ok=True)
CODE = Path('/content/medmnist-experiments')
print('BASE =', BASE)
'''

C_PLATFORM_KAGGLE = '''from pathlib import Path
import os, shutil, time
PLATFORM = "kaggle"

BASE = Path('/kaggle/working') / 'medmnist_benchmark'
SESSION_MARKER = Path('/kaggle/working/.medmnist-session-start')
if not SESSION_MARKER.exists():
    SESSION_MARKER.write_text(str(time.time()))
SESSION_START = float(SESSION_MARKER.read_text())

def _looks_like_medmnist_state(candidate):
    """Recognize both old nested exports and Kaggle datasets with files at root."""
    return (candidate.is_dir()
            and (candidate / 'runs').is_dir()
            and ((candidate / 'checkpoints').is_dir() or (candidate / 'state').is_dir()))

previous = set()
for attached in Path('/kaggle/input').iterdir():
    for candidate in (attached, attached / DRIVE_FOLDER):
        if ((candidate.name == DRIVE_FOLDER or candidate == attached)
                and _looks_like_medmnist_state(candidate)):
            previous.add(candidate.resolve())
previous = sorted(previous)
if len(previous) > 1:
    raise RuntimeError('expected zero or one MedMNIST state input; detach stale/duplicate inputs: '
                       + ', '.join(map(str, previous)))
if previous and not (BASE / 'runs').exists():
    shutil.copytree(previous[0], BASE, dirs_exist_ok=True)
    print('restored previous session state from', previous[0])
    print('restored checkpoints:', len(list((BASE / 'checkpoints').glob('*.pt')))
          if (BASE / 'checkpoints').is_dir() else 0)
for sub in ['runs', 'work', 'checkpoints', 'state']:
    (BASE / sub).mkdir(parents=True, exist_ok=True)
CODE = Path('/kaggle/working/medmnist-experiments')
print('BASE =', BASE)
'''

C_INSTALL = '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "medmnist==3.0.2", "tensorboardX", "acsconv", "scikit-learn"], check=True)
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "Enable a GPU runtime first."
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(BASE / "state" / "pip-freeze.txt").write_text(freeze)
print("environment frozen to", BASE / "state" / "pip-freeze.txt")
'''

C_CLONE = '''import shutil, subprocess
if CODE.exists() and not (CODE / ".git").exists():
    shutil.rmtree(CODE)
if not (CODE / ".git").exists():
    subprocess.run(["git", "clone", "--quiet", UPSTREAM_REPO, str(CODE)], check=True)
subprocess.run(["git", "-C", str(CODE), "checkout", "--quiet", UPSTREAM_COMMIT], check=True)
head = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
assert head == UPSTREAM_COMMIT, f"unexpected commit: {head}"
print("official source pinned at", head)
'''

C_PATCH_TAIL = '''
for _dimension in DIMENSIONS_THIS_SESSION:
    DIM_DIR = "MedMNIST2D" if _dimension == "2d" else "MedMNIST3D"
    TARGET = CODE / DIM_DIR / "train_and_eval_pytorch.py"
    _relative = f"{DIM_DIR}/train_and_eval_pytorch.py"
    _src = subprocess.check_output(["git", "-C", str(CODE), "show",
                                    f"{UPSTREAM_COMMIT}:{_relative}"], text=True)
    TARGET.write_text(patch_source(_src, _dimension), encoding="utf-8")
    compile(TARGET.read_text(encoding="utf-8"), str(TARGET), "exec")
    print("patched official script:", TARGET)
(CODE / "LOCAL_PATCHES.md").write_text(LOCAL_PATCHES_MD, encoding="utf-8")
'''

C_QUEUE = '''import json, re, time
PLAN_RUNS = json.loads(r\'\'\'QUEUE_PLACEHOLDER\'\'\')

def run_dir(run):
    return BASE / "runs" / run["run_id"]

def read_status(run):
    p = run_dir(run) / "status.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def is_done(run):
    st = read_status(run)
    return st is not None and st.get("status") == "completed" and st.get("verified") is True

def next_pending(dimension):
    for run in PLAN_RUNS:
        if run["dimension"] == dimension and not is_done(run):
            return run
    return None

pending = [r for r in PLAN_RUNS
           if r["dimension"] in DIMENSIONS_THIS_SESSION and not is_done(r)]
print(f"{len(pending)} pending runs in {DIMENSIONS_THIS_SESSION}; next:",
      pending[0]["run_id"] if pending else "none")
'''

C_LAUNCH = '''import torch
import threading
import medmnist

# One official run per physical GPU. The child process only ever sees its own GPU
# through CUDA_VISIBLE_DEVICES (set by child_environment), so --gpu_ids stays 0 and
# the official batch size / BatchNorm statistics are untouched.
LAUNCHER_LOCK = threading.Lock()
DOWNLOAD_LOCK = threading.Lock()

def ensure_dataset_downloaded(run):
    """Avoid two workers downloading/writing the same MedMNIST NPZ at once."""
    with DOWNLOAD_LOCK:
        info = medmnist.INFO[run["dataset"]]
        DataClass = getattr(medmnist, info["python_class"])
        DataClass(split="train", download=True, size=28)
        print("dataset ready:", run["dataset"])

def session_deadline():
    """Wall-clock instant at which training must pause at an epoch boundary."""
    return SESSION_START + MAX_SESSION_MINUTES * 60 - FINALIZE_MARGIN_SECONDS

def read_progress(ckpt):
    p = Path(str(ckpt) + ".progress.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def launch(run, gpu_index=0):
    rd = run_dir(run)
    rd.mkdir(parents=True, exist_ok=True)
    ckpt = BASE / "checkpoints" / (run["run_id"] + ".pt")
    dim_dir = "MedMNIST2D" if run["dimension"] == "2d" else "MedMNIST3D"
    script = CODE / dim_dir / "train_and_eval_pytorch.py"
    out_root = BASE / "work" / "output" / run["run_id"]
    deadline = session_deadline()
    cmd = [sys.executable, str(script),
           "--data_flag", run["dataset"], "--output_root", str(out_root),
           "--num_epochs", str(EPOCHS), "--size", "28", "--gpu_ids", "0",
           "--batch_size", str(run["batch_size"]), "--model_flag", "resnet50",
           "--run", run["run_id"], "--seed", str(run["seed"]),
           "--checkpoint", str(ckpt), "--resume", "--download"]
    if run["dimension"] == "2d":
        cmd += ["--resize", "--as_rgb"]
    else:
        cmd += ["--conv", "Conv3d"]
    resumed = ckpt.exists()
    status = {"run_id": run["run_id"], "dataset": run["dataset"], "seed": run["seed"],
              "dimension": run["dimension"], "model": "resnet50", "epochs": EPOCHS,
              "batch_size": run["batch_size"], "commit": UPSTREAM_COMMIT,
              "gpu_index": gpu_index, "resumed_from_checkpoint": resumed,
              "status": "running",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "verified": False}
    if "prior_attempts" in run:
        status["prior_attempts"] = run["prior_attempts"]
    with LAUNCHER_LOCK:
        write_json(rd / "status.json", status)
        write_json(rd / "command.json", cmd)
    ensure_dataset_downloaded(run)
    env = child_environment(gpu_index, run["seed"], deadline)
    with open(rd / "train.log", "a", encoding="utf-8") as lf:
        lf.write("CMD " + " ".join(cmd) + "\\n")
        lf.write("GPU_INDEX %d RESUMED %s DEADLINE %.0f\\n" % (gpu_index, resumed, deadline))
        lf.flush()
        code, timed_out = run_process(cmd, cwd=str(BASE / "work"), env=env, log=lf,
                                      deadline=deadline, poll_seconds=30)
    status["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status["exit_code"] = code
    status["timed_out"] = timed_out
    status.update(read_progress(ckpt))
    checkpoint_state = None
    completed_epochs = 0
    if ckpt.exists():
        try:
            checkpoint_state = torch.load(ckpt, map_location="cpu", weights_only=False)
            completed_epochs = checkpoint_completed_epochs(checkpoint_state)
            status["completed_epochs"] = completed_epochs
        except (OSError, RuntimeError, ValueError, KeyError) as err:
            status["status"] = "failed"
            status["note"] = "checkpoint validation failed: " + str(err)
            write_json(rd / "status.json", status)
            return status
    # Paused cleanly at an epoch boundary: the checkpoint stays, the next session resumes.
    # This also catches the training patch's intentional code-0 return before the deadline.
    if ckpt.exists() and is_resumable_pause(code, timed_out, completed_epochs, EPOCHS):
        status["status"] = "paused"
        status["note"] = ("session budget reached; atomic epoch checkpoint saved - "
                          "the next session resumes from completed_epochs")
        write_json(rd / "status.json", status)
        print(run["run_id"], "PAUSED at epoch", status.get("completed_epochs"), "- resumable")
        return status
    if code != 0:
        status["status"] = "failed"
        status["note"] = "non-zero exit; see train.log before rerunning"
        write_json(rd / "status.json", status)
        print(run["run_id"], "FAILED, exit", code, "- rerun the loop cell to resume")
        return status

    if checkpoint_state is None:
        status["status"] = "failed"
        status["note"] = "training exited without a checkpoint"
        write_json(rd / "status.json", status)
        return status
    resolved = Path(checkpoint_state["output_root_resolved"]).resolve()
    expected_output_root = (BASE / "work" / "output" / run["run_id"]).resolve()
    try:
        resolved.relative_to(expected_output_root)
    except ValueError:
        status["status"] = "failed"
        status["note"] = "checkpoint output path is outside this run's isolated work directory"
        write_json(rd / "status.json", status)
        return status
    del checkpoint_state
    try:
        # Independent re-check: recompute every split from the saved prediction CSVs
        # with the official evaluator and require agreement with the final log line.
        checked = verify_artifacts(run, resolved, completed_epochs, EPOCHS)
    except ValueError as err:
        status["status"] = "failed"
        status["note"] = "artifact verification refused this run: " + str(err)
        write_json(rd / "status.json", status)
        print(run["run_id"], "NOT VERIFIED:", err)
        return status
    status.update(status="completed", test_auc=checked["test_auc"], test_acc=checked["test_acc"],
                  verified=True, verification_method=checked["verification_method"],
                  artifact_sha256=checked["artifact_sha256"])
    for item in resolved.iterdir():
        if item.is_dir():
            shutil.copytree(item, rd / item.name, dirs_exist_ok=True)
        elif KEEP_BEST_MODEL or item.name != "best_model.pth":
            shutil.copy2(item, rd / item.name)
    # Make completion durable first. An orphaned checkpoint is harmless; a deleted
    # checkpoint with a stale running status would force a complete retrain.
    write_json(rd / "status.json", status)
    if status.get("verified") and ckpt.exists():
        ckpt.unlink()  # resume checkpoint no longer needed once a run is verified
        progress = Path(str(ckpt) + ".progress.json")
        if progress.exists():
            progress.unlink()
    if resolved.exists():
        shutil.rmtree(resolved)
    print(run["run_id"], status["status"], "verified:", status["verified"])
    return status
'''

C_MAIN = '''import time as _time
import torch as _torch
import threading as _threading
from concurrent.futures import ThreadPoolExecutor

# One official run per physical GPU, run as independent processes with
# CUDA_VISIBLE_DEVICES isolation. This is deliberately NOT DataParallel / DDP:
# splitting a single model across GPUs would change the effective batch size and
# BatchNorm statistics and would no longer reproduce the published baselines.

if not _torch.cuda.is_available():
    raise SystemExit("CUDA is False. Runtime -> Change runtime type -> T4 GPU -> Save, then Run all.")

GPU_COUNT = _torch.cuda.device_count()
WORKERS = max(1, min(PARALLEL_RUNS, GPU_COUNT))

print("GPUs visible:", GPU_COUNT, [(_torch.cuda.get_device_name(i)) for i in range(GPU_COUNT)])
print("workers (one run per GPU):", WORKERS, "| PARALLEL_RUNS =", PARALLEL_RUNS)
print("budget: MAX_SESSION_MINUTES =", MAX_SESSION_MINUTES, "| MIN_RUN_MINUTES =", MIN_RUN_MINUTES)
print("STOP_ON_FAILURE =", STOP_ON_FAILURE, "| dimensions:", DIMENSIONS_THIS_SESSION,
      "| scope:", DATASETS_THIS_SESSION)
if EXCLUDED_RUN_IDS:
    print("excluded run_ids (owned by another session):", EXCLUDED_RUN_IDS)

def minutes_left():
    return MAX_SESSION_MINUTES - (_time.time() - SESSION_START) / 60.0

_CLAIMED = set()
_CLAIM_LOCK = _threading.Lock()

def claim_next():
    """Reserve the next eligible run_id atomically, so no GPU ever races another."""
    with _CLAIM_LOCK:
        for run in PLAN_RUNS:
            if run["dimension"] not in DIMENSIONS_THIS_SESSION:
                continue
            if run["run_id"] in _CLAIMED or run["run_id"] in EXCLUDED_RUN_IDS:
                continue
            if run["dataset"] not in DATASETS_THIS_SESSION:
                continue
            if is_done(run):
                continue
            _CLAIMED.add(run["run_id"])
            return run
        return None

def worker(gpu_index):
    finished = []
    while minutes_left() > MIN_RUN_MINUTES:
        run = claim_next()
        if run is None:
            print(f"[gpu {gpu_index}] no eligible runs remain in scope")
            break
        print(f"=== START {run['run_id']} on gpu {gpu_index} ({minutes_left():.0f} min left) ===")
        status = launch(run, gpu_index=gpu_index)
        finished.append(status)
        print(f"=== END {run['run_id']}: status={status.get('status')} "
              f"verified={status.get('verified')} epochs={status.get('completed_epochs')} ===")
        if status.get("status") == "paused":
            print(f"[gpu {gpu_index}] session budget reached; stopping this worker")
            break
        if status.get("status") != "completed" or not status.get("verified"):
            if STOP_ON_FAILURE:
                print(f"[gpu {gpu_index}] stopping after a non-verified run; fix, then Run all to resume")
                break
    else:
        print(f"[gpu {gpu_index}] stopping: {minutes_left():.0f} min of budget remain "
              f"(< MIN_RUN_MINUTES={MIN_RUN_MINUTES})")
    return finished

results = []
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for part in pool.map(worker, range(WORKERS)):
        results.extend(part)

print(json.dumps([{"run_id": s["run_id"], "status": s["status"],
                   "completed_epochs": s.get("completed_epochs"),
                   "test_auc": s.get("test_auc"), "test_acc": s.get("test_acc"),
                   "verified": s["verified"]} for s in results], indent=2))
print(f"session used {minutes_left()*-1 + MAX_SESSION_MINUTES:.0f} of {MAX_SESSION_MINUTES} minutes")
if PLATFORM == "kaggle":
    print("Kaggle committed-version run finished; /kaggle/working will persist automatically.")
else:
    print("Drive-backed checkpoints and status files are already persistent.")
'''

C_SUMMARY = '''rows = []
for run in PLAN_RUNS:
    st = read_status(run) or {}
    verified = st.get("status") == "completed" and st.get("verified") is True
    rows.append({"run_id": run["run_id"], "dimension": run["dimension"],
                 "dataset": run["dataset"], "seed": run["seed"],
                 "status": st.get("status", "planned"),
                 "auc": st.get("test_auc") if verified else None,
                 "acc": st.get("test_acc") if verified else None,
                 "verified": verified, "reported_auc": run["reported_auc"],
                 "reported_acc": run["reported_acc"]})
done = [r for r in rows if r["status"] == "completed" and r["verified"]]
print(f"verified official runs: {len(done)}/90")
import pandas as pd
run_table = pd.DataFrame(rows)

summary_rows = []
for (dimension, dataset), planned_group in run_table.groupby(["dimension", "dataset"], sort=False):
    group = planned_group[planned_group["verified"]]
    reference_auc = float(planned_group["reported_auc"].iloc[0])
    reference_acc = float(planned_group["reported_acc"].iloc[0])
    summary = {"dimension": dimension, "dataset": dataset,
               "verified_runs": len(group), "expected_runs": 5,
               "reported_auc": reference_auc, "reported_acc": reference_acc,
               "mean_auc": None, "std_auc": None, "auc_difference": None,
               "auc_loss": None, "auc_loss_under_1pct": None,
               "mean_acc": None, "std_acc": None, "acc_difference": None,
               "acc_loss": None, "acc_loss_under_1pct": None,
               "all_verified": len(group) == 5}
    if len(group) == 5:
        mean_auc, mean_acc = float(group["auc"].mean()), float(group["acc"].mean())
        auc_loss, acc_loss = max(0.0, reference_auc - mean_auc), max(0.0, reference_acc - mean_acc)
        summary.update(mean_auc=mean_auc, std_auc=float(group["auc"].std(ddof=1)),
                       auc_difference=mean_auc - reference_auc, auc_loss=auc_loss,
                       auc_loss_under_1pct=auc_loss < 0.01,
                       mean_acc=mean_acc, std_acc=float(group["acc"].std(ddof=1)),
                       acc_difference=mean_acc - reference_acc, acc_loss=acc_loss,
                       acc_loss_under_1pct=acc_loss < 0.01)
    summary_rows.append(summary)

summary_table = pd.DataFrame(summary_rows)
reports = BASE / "reports"
reports.mkdir(parents=True, exist_ok=True)
for table, filename in ((run_table, "run-results.csv"),
                        (summary_table, "dataset-summary.csv")):
    target = reports / filename
    temporary = target.with_suffix(target.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, target)
print("wrote verified progress tables to", reports)
display(summary_table)
'''

MD_FOOTER = '''## Integrity checklist (every run)

- Same official protocol: ResNet-50, 100 epochs, official data pipeline and evaluator.
- Seed recorded in `status.json`; checkpoint path fixed per run id.
- Metrics come only from the official evaluator output and final log.
- `verified` is set only when the evaluator CSV exists and matches the parsed log.
- Failed runs stay in `runs/<run_id>/` with their log; they are never deleted or hidden.
'''


def _patch_cell_source():
    patch_3d = inspect.getsource(experiments.patch_3d_checkpoint)
    module_src = Path(patch_official.__file__).read_text(encoding='utf-8')
    module_src = module_src.replace('from __future__ import annotations\n', '')
    safety_src = Path(runner_safety.__file__).read_text(encoding='utf-8')
    safety_src = safety_src.replace('"""Tested reliability helpers embedded verbatim '
                                    'into the cloud notebooks."""\n', '')
    return ('# ==== Documented compatibility patches (verbatim from the tested project repo) ====\n'
            + patch_3d + '\n' + module_src + '\n'
            + '# ==== Tested reliability helpers (verbatim from scripts/runner_safety.py) ====\n'
            + safety_src + C_PATCH_TAIL)


def _code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.splitlines(keepends=True)}


def _md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


DUAL_OVERRIDES = {
    # Finish the benchmark ChestMNIST method gate first. The original session
    # owns seed 17; on T4x2 this starts seeds 29 and 43 as independent runs.
    'DATASETS_THIS_SESSION': '["chestmnist"]',
    'EXCLUDED_RUN_IDS': '["2d-chestmnist-resnet50-224-seed17-r1"]',
}

FINAL_OVERRIDES = {
    'DIMENSIONS_THIS_SESSION': '["2d", "3d"]',
    'DATASETS_THIS_SESSION': json.dumps([
        item['flag'] for item in
        json.loads((ROOT / 'configs/experiment-plan.json').read_text(encoding='utf-8'))['datasets']
    ]),
    'EXCLUDED_RUN_IDS': '[]',
    'KEEP_BEST_MODEL': 'True',
}

FINAL_NOTEBOOK_NAME = 'Habib_Ur_Rehman_MedMNIST_ResumeSafe.ipynb'


def apply_config_overrides(config, overrides):
    """Rewrite whole experiment statements in the configuration cell."""
    for name, value in (overrides or {}).items():
        pattern = rf'^{re.escape(name)} = .*(?:\n(?![A-Z_]+ =).*)*'
        match = re.search(pattern, config, flags=re.M)
        if match is None:
            raise ValueError(f'configuration key not found: {name}')
        config = config[:match.start()] + f'{name} = {value}' + config[match.end():]
    return config


def build_notebook(platform, config_overrides=None):
    if platform not in ('colab', 'kaggle'):
        raise ValueError("platform must be 'colab' or 'kaggle'")
    platform_cell = C_PLATFORM_COLAB if platform == 'colab' else C_PLATFORM_KAGGLE
    config = apply_config_overrides(C_CONFIG.replace('COMMIT_PLACEHOLDER', COMMIT),
                                    config_overrides)
    cells = [
        _md(_header_md(platform)),
        _md('## 1. Configuration'),
        _code(config),
        _md('## 2. Persistent storage'),
        _code(platform_cell),
        _md('## 3. Dependencies (environment is frozen after this cell)'),
        _code(C_INSTALL),
        _md('## 4. Official source, pinned and verified'),
        _code(C_CLONE),
        _md('## 5. Documented compatibility patches\n\n'
            'Seeding, checkpoint/resume, and the 3D best-checkpoint fix. '
            'Training math and evaluation are untouched; see `LOCAL_PATCHES.md`.'),
        _code(_patch_cell_source()),
        _md('## 6. Preregistered run queue'),
        _code(C_QUEUE.replace('QUEUE_PLACEHOLDER', embedded_queue_json())),
        _md('## 7. Launcher (resume-safe, artifact-verified)'),
        _code(C_LAUNCH),
        _md('## 8. Process this session'),
        _code(C_MAIN),
        _md('## 9. Progress summary'),
        _code(C_SUMMARY),
        _md(MD_FOOTER),
    ]
    for index, cell in enumerate(cells):
        cell['id'] = f'cell-{index:02d}'
    metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }
    if platform == 'colab':
        metadata['accelerator'] = 'GPU'
        metadata['colab'] = {'provenance': [], 'gpuType': 'T4'}
    else:
        metadata['kaggle'] = {'accelerator': 'gpu', 'isInternetEnabled': True,
                              'isGpuEnabled': True}
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': metadata, 'cells': cells}


def build_all(out_dir=ROOT / 'notebooks'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    variants = [('colab', 'colab_medmnist_runner.ipynb', None),
                ('kaggle', 'kaggle_medmnist_runner.ipynb', None),
                ('kaggle', 'kaggle_medmnist_runner_dual.ipynb', DUAL_OVERRIDES)]
    for platform, name, overrides in variants:
        path = out_dir / name
        path.write_text(json.dumps(build_notebook(platform, overrides), indent=1) + '\n',
                        encoding='utf-8')
        paths.append(path)
    return paths


def build_final_kaggle_notebook(out_dir):
    """Write the single upload-ready resume-safe notebook."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FINAL_NOTEBOOK_NAME
    path.write_text(json.dumps(build_notebook('kaggle', FINAL_OVERRIDES), indent=1) + '\n',
                    encoding='utf-8')
    return path


if __name__ == '__main__':
    for path in build_all():
        print('wrote', path)


