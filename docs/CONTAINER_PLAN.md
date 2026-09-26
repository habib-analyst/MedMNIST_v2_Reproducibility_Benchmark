# Container plan (not yet built or tested)

A container is a planned deliverable, not a completed artifact. When GPU access is available, the container should be built and tested in this order:

1. Start from a pinned NVIDIA CUDA base image with Python 3.12.
2. Install `requirements.txt`, then verify the resolved environment against the relevant evidence record.
3. Fetch the pinned upstream source: `https://github.com/MedMNIST/experiments` at commit `70b6b3a7ad7afddff1df2a3b735235830fbdb142`.
4. Run a CPU smoke test.
5. Run one 3D calibration job and record GPU model, peak CUDA memory, input shape, batch size, and seconds per epoch.
6. Run the Phase 1 schedule with checkpoint/resume enabled.
7. Publish the image digest, run manifest, and checksums with the final artifacts.

No container image digest is claimed in this repository because no tested image has been built yet.
