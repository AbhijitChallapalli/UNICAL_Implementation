# HPC execution notes

The Python entrypoints are scheduler-independent. Existing Slurm files in the working directory contain cluster-specific partitions, exclusions, log locations, and historical run settings, so they are intentionally not part of the proposed public tree.

`setup_hpc.example.sh` and `common.example.sh` are portable starting points derived from the historical helpers. Copy and review them for the target cluster before use. The published computations used Python 3.10.20, CUDA 12.8, and an NVIDIA H100 80 GB GPU; the setup example currently installs PyTorch 2.7.1 from a CUDA 12.6 wheel index and creates Python 3.11 when its environment is absent. It is therefore not an exact environment lock for the published run.
