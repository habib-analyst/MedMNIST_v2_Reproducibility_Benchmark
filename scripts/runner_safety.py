"""Tested reliability helpers embedded verbatim into the cloud notebooks."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def child_environment(gpu_index, seed, deadline):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu_index), PYTHONHASHSEED=str(seed),
               MEDMNIST_DEADLINE=str(deadline), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def select_session_runs(queue, dimension, datasets, excluded, is_done, limit):
    selected = []
    seen = set(excluded)
    for run in queue:
        if (run['dimension'] != dimension or run['dataset'] not in datasets
                or run['run_id'] in seen or is_done(run)):
            continue
        selected.append(run)
        seen.add(run['run_id'])
        if len(selected) >= limit:
            break
    return selected


def run_process(command, cwd, env, log, deadline, poll_seconds=30):
    """Hard backstop; normal pause happens after an atomic epoch checkpoint."""
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT)
    timed_out = False
    try:
        while process.poll() is None:
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                process.terminate()
                break
            try:
                process.wait(timeout=min(poll_seconds, remaining))
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        process.wait()
    return process.returncode, timed_out


def is_resumable_pause(exit_code, timed_out, completed_epochs, expected_epochs):
    """True only when a real epoch checkpoint can safely continue later."""
    return (0 < completed_epochs < expected_epochs
            and (timed_out or exit_code == 0))


def checkpoint_completed_epochs(state):
    """The atomic checkpoint is authoritative; progress JSON is informational."""
    epoch = state.get('epoch')
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError('checkpoint has no valid completed epoch')
    return epoch + 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(run, resolved, completed_epochs, expected_epochs,
                     evaluator_factory=None):
    """Recompute all split metrics with official labels, without rewriting CSVs."""
    import csv
    import numpy as np
    if evaluator_factory is None:
        from medmnist import Evaluator
        evaluator_factory = Evaluator
    if completed_epochs != expected_epochs:
        raise ValueError('training epochs incomplete or incompatible')
    resolved = Path(resolved)
    log_path = resolved / (run['dataset'] + '_log.txt')
    model_path = resolved / 'best_model.pth'
    if not log_path.is_file():
        raise ValueError('final evaluator log is missing')
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise ValueError('best-validation model is missing')
    logs = log_path.read_text(encoding='utf-8')
    hashes = {p.name: file_sha256(p) for p in (log_path, model_path)}
    verified = {}
    for split in ('train', 'val', 'test'):
        files = list(resolved.glob(f"{run['dataset']}_{split}_*@{run['run_id']}.csv"))
        if len(files) != 1:
            raise ValueError(f'{split}: expected exactly one prediction CSV')
        with files[0].open(newline='', encoding='utf-8') as stream:
            rows = [row for row in csv.reader(stream) if row]
        table = np.asarray([[float(cell) for cell in row] for row in rows])
        index = table[:, 0].astype(np.int64)
        scores = table[:, 1:]
        evaluator = evaluator_factory(run['dataset'], split, size=28)
        n = len(evaluator.labels)
        if len(scores) != n or not np.array_equal(index, np.arange(n)):
            raise ValueError(f'{split}: prediction rows or ordering disagree with official labels')
        expected_columns = len(evaluator.info['label'])
        if scores.shape != (n, expected_columns):
            raise ValueError(f'{split}: incorrect prediction shape')
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError(f'{split}: invalid prediction probabilities')
        auc, acc = evaluator.evaluate(scores)
        matches = re.findall(rf'{split}  auc: ([0-9.]+)  acc: ([0-9.]+)', logs)
        if (not matches or not np.isfinite([auc, acc]).all()
                or not np.allclose([auc, acc], [float(x) for x in matches[-1]],
                                   rtol=0, atol=0.0000051)):
            raise ValueError(f'{split}: recomputed metrics disagree with final log')
        expected_name = f"{run['dataset']}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run['run_id']}.csv"
        if files[0].name != expected_name:
            raise ValueError(f'{split}: recomputed metrics disagree with CSV name')
        verified[split + '_auc'], verified[split + '_acc'] = float(auc), float(acc)
        hashes[files[0].name] = file_sha256(files[0])
    return dict(verified, artifact_sha256=hashes,
                verification_method='official-evaluator-recomputed-from-predictions')
