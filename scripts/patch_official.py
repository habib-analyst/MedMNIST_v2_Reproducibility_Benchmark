"""Documented compatibility patches for the pinned official MedMNIST experiment scripts.

Reliability and reproducibility only. Training math, model definitions, data
pipelines, optimizer, and evaluation are untouched.

Patches:
1. ``--seed``: deterministic seeding (upstream sets no seed at all).
2. ``--checkpoint`` / ``--resume``: end-of-epoch checkpoint (model, optimizer,
   scheduler, best model, epoch, RNG states, resolved output directory) and
   resume from it. Required because free cloud GPU sessions disconnect;
   without resume a disconnect discards a multi-hour official run.
3. (3D only) save the best-validation model instead of the final-epoch model,
   matching what upstream evaluates (see ``scripts/experiments.py``).

RNG states are restored after initialization. Exact cross-device or cross-version
numerical equivalence is not guaranteed; environment changes must be disclosed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SEED_BLOCK = '''
import random


def seed_everything(seed):
    import os
    import numpy as np
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
'''

RESUME_BLOCK = '''
    start_epoch = 0
    if resume and checkpoint is not None and os.path.exists(checkpoint):
        # Checkpoint holds RNG/scheduler state, not tensors only; it is written by this runner.
        _ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(_ckpt['model'])
        optimizer.load_state_dict(_ckpt['optimizer'])
        scheduler.load_state_dict(_ckpt['scheduler'])
        best_model.load_state_dict(_ckpt['best_model'])
        best_auc = _ckpt['best_auc']
        best_epoch = _ckpt['best_epoch']
        start_epoch = _ckpt['epoch'] + 1
        iteration = _ckpt['iteration']
        random.setstate(_ckpt['python_rng'])
        np.random.set_state(_ckpt['numpy_rng'])
        torch.set_rng_state(_ckpt['torch_rng'])
        if torch.cuda.is_available() and _ckpt['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(_ckpt['cuda_rng'])
        print('==> Resumed from checkpoint at epoch', start_epoch, flush=True)
        del _ckpt
'''

CHECKPOINT_BLOCK = '''
        if checkpoint is not None:
            _ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_model': best_model.state_dict(),
                'best_auc': best_auc,
                'best_epoch': best_epoch,
                'epoch': epoch,
                'iteration': iteration,
                'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'output_root_resolved': output_root,
            }
            _tmp = checkpoint + '.tmp'
            with open(_tmp, 'wb') as _stream:
                torch.save(_ckpt, _stream)
                _stream.flush()
                os.fsync(_stream.fileno())
            os.replace(_tmp, checkpoint)
            _elapsed = time.time() - _epoch_started
            print('CHECKPOINT completed_epochs=%d epoch_seconds=%.1f' %
                  (epoch + 1, _elapsed), flush=True)
            import json
            _progress = {'completed_epochs': epoch + 1, 'total_epochs': num_epochs,
                         'last_epoch_seconds': _elapsed,
                         'remaining_estimate_hours': (num_epochs - epoch - 1) * _elapsed / 3600}
            with open(checkpoint + '.progress.json.tmp', 'w') as _stream:
                json.dump(_progress, _stream)
            os.replace(checkpoint + '.progress.json.tmp', checkpoint + '.progress.json')
            _deadline = float(os.environ.get('MEDMNIST_DEADLINE', 'inf'))
            if epoch + 1 < num_epochs and time.time() + max(60, _elapsed * 1.5) + 900 >= _deadline:
                print('PAUSED: checkpoint saved; insufficient budget for another epoch', flush=True)
                writer.close()
                return
'''

OUTPUT_ROOT_OLD = '''    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)'''

OUTPUT_ROOT_NEW = '''    if resume and checkpoint is not None and os.path.exists(checkpoint):
        _meta = torch.load(checkpoint, map_location='cpu', weights_only=False)
        output_root = _meta['output_root_resolved']
    else:
        output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)'''

ARGPARSE_OLD = "    parser.add_argument('--run',"
ARGPARSE_NEW = '''    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run','''

_SIGNATURES = {
    '2d': "def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run):",
    '3d': "def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run):",
}

_CALLS = {
    '2d': "main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run)",
    '3d': "main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run)",
}

LOCAL_PATCHES_MD = """# Local compatibility patches

Official code pinned from https://github.com/MedMNIST/experiments
(commit 70b6b3a7ad7afddff1df2a3b735235830fbdb142). Training math, model
definitions, data pipelines, optimizer, and evaluation are untouched.

1. `--seed`: deterministic seeding; upstream sets no seed.
2. `--checkpoint` / `--resume`: end-of-epoch checkpoint and resume so a
   disconnected free-GPU session does not discard a multi-hour official run.
   RNG states are restored on the CPU after initialization. Cross-device and
   cross-version bitwise reproducibility is not guaranteed. Epoch checkpoints
   use flush/fsync and atomic replacement. A session deadline pauses at an
   epoch boundary; a launcher watchdog handles a stalled epoch.
3. (3D only) `best_model.pth` now stores the best-validation model that
   upstream evaluates; upstream saved the final-epoch model under that name.
4. Respect the launcher's GPU visibility so independent seeds use separate
   GPUs with the original batch size, not DataParallel or distributed training.
"""


def _load_3d_checkpoint_patch():
    existing = globals().get('patch_3d_checkpoint')
    if existing is not None:
        return existing
    try:
        from experiments import patch_3d_checkpoint as fn
        return fn
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            'experiments', Path(__file__).with_name('experiments.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.patch_3d_checkpoint


def _replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f'patch anchor must appear exactly once: {label}')
    return source.replace(old, new, 1)


def patch_source(source, dimension):
    """Apply the documented patches to an official training script source."""
    if dimension not in ('2d', '3d'):
        raise ValueError("dimension must be '2d' or '3d'")
    if 'start_epoch' in source or '--seed' in source:
        raise ValueError('source is already patched')
    if dimension == '3d':
        source = _load_3d_checkpoint_patch()(source)

    source = _replace_once(
        source, 'from tqdm import trange', 'from tqdm import trange' + SEED_BLOCK, 'trange import')

    signature = _SIGNATURES[dimension]
    source = _replace_once(
        source, signature,
        signature[:-2] + ', seed=None, checkpoint=None, resume=False):'
        + '\n    if seed is not None:\n        seed_everything(seed)\n', 'main signature')

    source = _replace_once(source, OUTPUT_ROOT_OLD, OUTPUT_ROOT_NEW, 'output root')
    source = _replace_once(
        source, 'os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])',
        'os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_ids[0]))', 'GPU visibility')

    source = _replace_once(
        source, '    global iteration\n    iteration = 0',
        '    global iteration\n    iteration = 0\n' + RESUME_BLOCK, 'resume block')

    source = _replace_once(
        source, '    for epoch in trange(num_epochs):',
        '    for epoch in trange(start_epoch, num_epochs):\n        _epoch_started = time.time()', 'loop start')

    source = _replace_once(
        source, "            print('cur_best_epoch', best_epoch)\n",
        "            print('cur_best_epoch', best_epoch)\n" + CHECKPOINT_BLOCK, 'checkpoint save')

    source = _replace_once(source, ARGPARSE_OLD, ARGPARSE_NEW, 'argparse args')

    call = _CALLS[dimension]
    source = _replace_once(
        source, call,
        call[:-1] + ', seed=args.seed, checkpoint=args.checkpoint, resume=args.resume)',
        'call site')
    return source
