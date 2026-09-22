---
name: launch-experiment
description: Use whenever a project experiment needs to be launched on CPU or GPU, locally or on ORCD with Slurm.
---

# Launch an experiment

1. Run `python -m unittest discover -s tests`, then `python experiment.py --device cpu` for a local check. Use `--device cuda` only when CUDA is available.
2. On ORCD, run training on a compute node with `sbatch`, never on a login node. Use `mit_quicktest` for short CPU runs; request a GPU on `mit_normal_gpu` only when needed. Load the project's PyTorch environment in the batch script and set a short time limit.
3. Check `squeue --me`, then inspect the job log and `sacct -j JOB_ID`. Report the job ID, device, exit state, and both models' test accuracies.
