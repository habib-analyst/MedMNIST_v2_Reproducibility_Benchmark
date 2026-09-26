# Installation and pinned environment

The retained environment freeze is copied to `requirements-lock.txt`. It records the packages installed in the historical GPU environment, including Python 3.12, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, and medmnist 3.0.2.

## Fresh-clone source setup

```bash
git clone https://github.com/MedMNIST/experiments.git external/medmnist-experiments
git clone https://github.com/MedMNIST/MedMNIST.git external/medmnist
cd external/medmnist-experiments
git checkout 70b6b3a7ad7afddff1df2a3b735235830fbdb142
cd ../..
```

The scripts expect these exact directory names: `external/medmnist-experiments` holds the pinned experiments code; `external/medmnist` holds the MedMNIST package metadata used to build the run plan.

## Import layout

The `src/` package is imported directly by the scripts through a repository-root path setup (`sys.path` insertion in each script). No separate installation or packaging step is required; a installable `pyproject.toml` layout is planned once the baseline artifacts exist.

Install the frozen requirements in a clean environment:

```bash
python -m venv .venv
# activate the environment
python -m pip install -r requirements-lock.txt
```

The retained 2D evidence was produced by a Kaggle GPU kernel. The notebook in `kaggle/runner/` is a prepared 3D notebook and has not been executed. The local `scripts/run_official_2d.py` is a CPU pilot and is not the GPU evidence path.

## Container status

A tested container is not yet published. The final container must use a pinned CUDA base image, install `requirements-lock.txt`, fetch the pinned MedMNIST source, and document the GPU command. This is a planned deliverable, not a completed artifact.

