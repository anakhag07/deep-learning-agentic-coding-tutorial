"""Compare linear and nonlinear classifiers on synthetic concentric circles."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def save_plot(models, points, labels, results, path, device):
    import matplotlib

    matplotlib.use("Agg")  # Save figures on headless Slurm nodes.
    import matplotlib.pyplot as plt

    axis = torch.linspace(-1.4, 1.4, 160)
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    grid = torch.stack((xx.ravel(), yy.ravel()), dim=1).to(device)
    fig, panels = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, layout="constrained")
    for panel, (name, model) in zip(panels, models.items()):
        with torch.no_grad():
            probability = model(grid).sigmoid().reshape(xx.shape).cpu().numpy()
        panel.contourf(xx, yy, probability, levels=20, cmap="coolwarm", vmin=0, vmax=1)
        panel.contour(xx, yy, probability, levels=[0.5], colors="black", linewidths=1)
        panel.scatter(points[:, 0], points[:, 1], c=labels, cmap="coolwarm", vmin=0, vmax=1,
                      edgecolors="black", linewidths=0.2, s=15)
        title = "Linear" if name == "linear" else "MLP (ReLU)"
        panel.set(title=f"{title} · test acc {results[name]['test_accuracy']:.2f}",
                  xlabel="x₁", ylabel="x₂", aspect="equal")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(device="cpu", steps=200, seed=0, plot_path=None):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)
    n = 512
    labels = (torch.arange(n) % 2).float()
    angles = 2 * math.pi * torch.rand(n, generator=generator)
    radii = 0.4 + 0.6 * labels + 0.08 * torch.randn(n, generator=generator)
    points = torch.stack((radii * angles.cos(), radii * angles.sin()), dim=1)
    order = torch.randperm(n, generator=generator)
    points, labels = points[order].to(device), labels[order].to(device)
    train_x, test_x = points[:384], points[384:]
    train_y, test_y = labels[:384], labels[384:]

    linear = nn.Linear(2, 1)  # One straight decision boundary.
    mlp = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),  # Makes a curved boundary possible.
        nn.Linear(16, 1),
    )
    models = {"linear": linear, "mlp": mlp}
    results = {}
    for name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        def loss():
            return F.binary_cross_entropy_with_logits(model(train_x).squeeze(1), train_y)

        initial_loss = loss().item()
        for _ in range(steps):
            optimizer.zero_grad()
            loss().backward()
            optimizer.step()
        with torch.no_grad():
            accuracy = ((model(test_x).squeeze(1) > 0) == test_y.bool()).float().mean()
            results[name] = {
                "initial_loss": round(initial_loss, 3),
                "final_loss": round(loss().item(), 3),
                "test_accuracy": round(accuracy.item(), 3),
            }
    if plot_path:
        save_plot(models, test_x.cpu(), test_y.cpu(), results, plot_path, device)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--plot", default="outputs/decision_boundaries.png")
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.steps, plot_path=args.plot), indent=2))
    print(f"Plot: {args.plot}")
