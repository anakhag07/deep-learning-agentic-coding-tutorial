# 6.7960 agentic coding demo

Recreate `experiment.py` and `tests/test_experiment.py` when missing. Keep this standalone exercise separate from course problem sets and concise enough for a one-hour tutorial.

- `experiment.py`: generate 512 concentric-circle points (seed 0); train `nn.Linear(2, 1)` and `Linear(2, 16) → ReLU → Linear(16, 1)` on the same split for about 200 steps. Default to CPU, allow CUDA, print loss and held-out accuracy, and save side-by-side matplotlib decision boundaries with short titles and labeled axes to `outputs/decision_boundaries.png`.
- `tests/test_experiment.py`: one unittest showing that training lowers loss and MLP held-out accuracy exceeds 0.90 and the linear model by 0.15.
- Use only PyTorch, matplotlib, and the standard library; no downloads. Run `python -m unittest discover -s tests` and `python experiment.py --device cpu`.
- Use `launch-experiment` for every CPU/GPU experiment run and `visualize-experiment` for plots. On ORCD, train through Slurm on a compute node.
