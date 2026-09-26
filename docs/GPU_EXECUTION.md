# GPU execution

The Kaggle GPU runner is `kaggle/runner/medmnist_3d_suite_kaggle.ipynb`. It clones/uses the pinned MedMNIST source, prepares the registered datasets, and runs the configured seeds with per-epoch checkpoints.

The historical 2D run used the official script with GPU index 0 and batch size 128. The exact GPU model was not preserved in the evidence metadata and is therefore not claimed.

The first 3D run must be treated as a calibration run. It must record:

```text
GPU model
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
batch size
input dimensions
seconds per epoch
```

After calibration, the measured values should be written to `docs/RUNTIME_AND_VRAM.md` and used for scheduling the remaining runs.
