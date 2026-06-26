"""Vanilla ULA sampling (sec:discretization, eq:euler_maruyama).

The discrete overdamped Langevin chain (the Unadjusted Langevin Algorithm)

    x_{k+1} = x_k + eta * grad log p(x_k) + sqrt(2 eta) z_k,   z_k ~ N(0, I)

samples a 2-D target p. This is the foundational sampler the thesis rests on,
shown here on a plain two-mode target (no data / forward operator), one panel:

  * pale dots : the endpoints of many independent chains, reproducing p;
  * one chain : an example discrete walk on top (drift along the score plus
                isotropic noise), light -> dark over steps.

Self-contained: analytic two-mode GMM score + numpy ULA. Imports landscape only
for the shared visual language.

Run:  ./.venv/bin/python3 thesis_figures/fig_ula_sampling.py
"""

from __future__ import annotations

import argparse
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import landscape as L

MODES = np.array([[-1.5, 0.0], [1.5, 0.0]])
STD = 0.6
NB = 700                  # independent chains
K = 400                   # steps
ETA = 0.04                # step size
LIM = 3.4
CMAP = LinearSegmentedColormap.from_list(
    "cooling", ["#dfe7f0", "#a8bcd4", "#5a7aa3", L.SAMPLE])   # light/early -> dark/late


def score(x):
    """grad log p(x) for the equal-weight two-mode Gaussian mixture."""
    diff = MODES[None] - x[:, None]                  # (nb, K, 2)
    m = -0.5 * (diff ** 2).sum(-1) / STD ** 2        # (nb, K)
    w = np.exp(m - m.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * diff).sum(1) / STD ** 2


def ula(rng):
    x = rng.normal(0.0, 1.5, size=(NB, 2))
    x[0] = np.array([0.1, -2.5])         # example chain: a fixed, in-frame start
    path = [x[0].copy()]
    for _ in range(K):
        x = x + ETA * score(x) + np.sqrt(2 * ETA) * rng.normal(size=(NB, 2))
        path.append(x[0].copy())
    return x, np.array(path)


def density_grid(n=260):
    g = np.linspace(-LIM, LIM, n)
    xx, yy = np.meshgrid(g, g)
    p = sum(np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2 * STD ** 2))
            for mx, my in MODES)
    return xx, yy, p / p.max()


def filled_density(ax, grid):
    xx, yy, p = grid
    lv = np.quantile(p[p > p.max() * 1e-3], [0.5, 0.8, 0.95])
    ax.contourf(xx, yy, p, levels=np.r_[lv, p.max()], colors=L.DENSITY, zorder=0)


def frame(ax):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=13, length=3)
    ax.set_xlabel("$x_1$", fontsize=15)


def main():
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()
    L.apply_style()

    rng = np.random.default_rng(3)
    endpts, path = ula(rng)
    grid = density_grid()

    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    filled_density(ax, grid)
    ax.scatter(endpts[:, 0], endpts[:, 1], s=5, c=L.SAMPLE, alpha=0.20,
               linewidths=0, rasterized=True, zorder=3)
    seg = np.stack([path[:-1], path[1:]], axis=1)
    lc = LineCollection(seg, cmap=CMAP, array=np.linspace(0, 1, len(path) - 1),
                        linewidths=0.9, alpha=0.9, zorder=5)
    ax.add_collection(lc)
    ax.scatter(path[:, 0], path[:, 1], c=np.linspace(0, 1, len(path)), cmap=CMAP,
               s=7, zorder=6, linewidths=0)
    ax.plot([path[0, 0]], [path[0, 1]], "o", mfc="white", mec=L.SUBTLE, ms=7,
            mew=1.0, zorder=7)
    ax.annotate("start", xy=(path[0, 0], path[0, 1]), xytext=(7, -10),
                textcoords="offset points", fontsize=13, color=L.INK)
    ax.set_ylabel("$x_2$", fontsize=15)
    frame(ax)
    fig.text(0.012, 0.95, r"$x_{k+1} = x_k + \eta\,\nabla\log p(x_k) + "
             r"\sqrt{2\eta}\,z_k$  (drift along the score $+$ isotropic noise)",
             fontsize=14.5, color=L.INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=L.DENSITY[2], edgecolor="none", label=r"target $p(x)$"),
        Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=6, alpha=0.4,
               label=rf"chain endpoints"),
        Line2D([], [], color=L.SAMPLE, marker="o", ms=4, lw=1.4,
               label=r"one chain (light$\to$dark = steps)"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=L.SUBTLE, ms=8,
               label="start"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.925),
               ncol=4, frameon=False, fontsize=13, handletextpad=0.3,
               columnspacing=.6, labelcolor=L.INK, borderaxespad=0.0)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.10, right=0.98)

    out = f"{args.out_dir}/fig_ula_sampling"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
