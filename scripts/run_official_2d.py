
"""Launch one official MedMNIST2D ResNet-50 run with deterministic seeding."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_env(env_dir: Path) -> Path:
    src = ROOT / "external" / "medmnist-experiments" / "MedMNIST2D"
    if env_dir.exists():
        return env_dir
    shutil.copytree(src, env_dir)
    train = env_dir / "train_and_eval_pytorch.py"
    text = train.read_text(encoding="utf-8")
    needle = "def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run):"
    if "seed_everything" not in text:
        inject = (
            "\n\ndef seed_everything(seed):\n"
            "    import os, random\n"
            "    import numpy as np\n"
            "    import torch\n"
            "    os.environ['PYTHONHASHSEED']=str(seed)\n"
            "    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)\n"
            "    if torch.cuda.is_available():\n"
            "        torch.cuda.manual_seed_all(seed)\n"
            "    torch.backends.cudnn.deterministic=True\n"
            "    torch.backends.cudnn.benchmark=False\n"
        )
        text = text.replace("from tqdm import trange", "from tqdm import trange" + inject, 1)
        # add seed arg to main signature and first line body
        text = text.replace(
            needle,
            "def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run, seed=17):\n    seed_everything(seed)",
            1,
        )
        # argparse
        if "--seed" not in text:
            text = text.replace(
                "parser.add_argument('--run',\n",
                "parser.add_argument('--seed', default=17, type=int)\n    parser.add_argument('--run',\n",
                1,
            )
            text = text.replace(
                "run = args.run\n    \n    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run)",
                "run = args.run\n    seed = args.seed\n    \n    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, download, model_flag, resize, as_rgb, model_path, run, seed)",
                1,
            )
        train.write_text(text, encoding="utf-8")
    return env_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_flag", default="chestmnist")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--run_id", default=None)
    p.add_argument("--download", action="store_true", default=True)
    args = p.parse_args()

    run_id = args.run_id or f"2d-{args.data_flag}-resnet50-224-seed{args.seed}-r1"
    out = ROOT / "runs" / "official" / run_id
    out.mkdir(parents=True, exist_ok=True)
    env_dir = ensure_env(ROOT / "runs" / "_env_medmnist2d")

    status = {
        "kind": "official-2d",
        "run_id": run_id,
        "dataset": args.data_flag,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "model": "resnet50",
        "resolution": 224,
        "device": "cpu",
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command_note": "Official MedMNIST2D train_and_eval_pytorch with local seed injection; gpu_ids=-1 for CPU",
    }
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (out / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-c", "import torch,medmnist,sys; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(medmnist.__version__)"], text=True),
        encoding="utf-8",
    )

    seed_everything(args.seed)
    cmd = [
        sys.executable,
        str(env_dir / "train_and_eval_pytorch.py"),
        "--data_flag", args.data_flag,
        "--output_root", str(out / "output"),
        "--num_epochs", str(args.epochs),
        "--size", "28",
        "--gpu_ids", "-1",
        "--batch_size", str(args.batch_size),
        "--model_flag", "resnet50",
        "--resize",
        "--as_rgb",
        "--run", run_id,
        "--seed", str(args.seed),
    ]
    if args.download:
        cmd.append("--download")

    (out / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    log_path = out / "train.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("CMD " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(env_dir), stdout=log, stderr=subprocess.STDOUT, text=True)
    status["status"] = "completed" if proc.returncode == 0 else "failed"
    status["exit_code"] = proc.returncode
    status["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
