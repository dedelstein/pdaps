"""Preconditioning figure (the isotropic vs preconditioned INNER step, Sec. 4.5-4.6).

The stochastic twin of fig_gd_conditioning: sampling a stiff (anisotropic) Gaussian
under a fixed step budget, all chains started from the centre.

  * isotropic Langevin (the DPS/DAPS inner step) must keep the step small for
    stability on the stiff axis, so in a fixed budget it only UNDER-MIXES -- the
    cloud never spreads to fill the long axis of the target;
  * preconditioned Langevin (the pULA/pDAPS inner step, metric M = H^{-1}) equalises
    the curvature, permitting a larger step, and COVERS the target in the same budget.

On a single Gaussian DPS==DAPS and pULA==pDAPS (no modes), so this is a 2-way
comparison: it isolates the preconditioning column of the 2x2.

Run:  ./.venv/bin/python3 thesis_figures/fig_preconditioning.py
"""

from __future__ import annotations

import numpy as np

import landscape as L

S_LONG, S_SHORT = 2.4, 0.20                 # target std along the long / stiff axes
SIGMA = np.diag([S_LONG ** 2, S_SHORT ** 2])
H = np.diag([1.0 / S_LONG ** 2, 1.0 / S_SHORT ** 2])
MHALF = np.diag([S_LONG, S_SHORT])          # M^{1/2} for M = SIGMA = H^{-1}
LIM = 3.2
STEPS = 40
NB = 1500


def langevin(M, Mhalf, eta, T, nb, seed=0, keep_path=True):
    rng = np.random.default_rng(seed)
    x = np.zeros((nb, 2))
    path = [x[0].copy()] if keep_path else None
    MH = M @ H
    for _ in range(T):
        x = x - eta * (x @ MH.T) + np.sqrt(2 * eta) * (rng.standard_normal((nb, 2)) @ Mhalf.T)
        if keep_path:
            path.append(x[0].copy())
    return x, (np.array(path) if keep_path else None)


def gaussian_contours(ax):
    g = np.linspace(-LIM, LIM, 240)
    X, Y = np.meshgrid(g, g)
    Z = 0.5 * (X ** 2 / S_LONG ** 2 + Y ** 2 / S_SHORT ** 2)
    ax.contour(X, Y, Z, levels=np.linspace(0.5, 12, 6), colors=L.MODE_EDGE,
               linewidths=0.4, alpha=0.65)


def panel(ax, cloud, path, title, note):
    gaussian_contours(ax)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=2.0, c=L.SAMPLE, alpha=0.18,
               linewidths=0, rasterized=True)
    ax.plot(path[:, 0], path[:, 1], "-", color=L.SAMPLE, lw=0.7, alpha=0.9)
    ax.plot(path[:, 0], path[:, 1], ".", color=L.SAMPLE, ms=2.8, alpha=0.9)
    ax.plot([0], [0], "o", mfc="white", mec=L.SAMPLE, ms=7, mew=1.1, zorder=6)
    ax.set_title(title, fontsize=19, loc="left")
    ax.text(0.04, 0.05, note, transform=ax.transAxes, fontsize=15, color=L.INK,
            va="bottom")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=13, length=3)
    ax.set_xlabel("long (flat) axis  $x_1$", fontsize=16)


def main():
    import argparse
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()
    L.apply_style()

    eta_iso = 0.3 * S_SHORT ** 2                 # bounded by the stiff axis
    eta_pre = 0.25                                # the metric permits a far larger step
    iso_cloud, iso_path = langevin(np.eye(2), np.eye(2), eta_iso, STEPS, NB, seed=1)
    pre_cloud, pre_path = langevin(SIGMA, MHALF, eta_pre, STEPS, NB, seed=1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.9), sharey=True)
    panel(a1, iso_cloud, iso_path, "isotropic Langevin",
          "small step (stiff-axis bound):\nunder-explores the long axis")
    panel(a2, pre_cloud, pre_path, "preconditioned Langevin",
          "equalised, larger step:\ncovers the target")
    a1.set_ylabel("stiff axis  $x_2$", fontsize=16)
    fig.text(0.012, 0.955, rf"{NB} chains, {STEPS} steps from the centre "
             "(fixed budget)", fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="white", mec=L.SAMPLE, ms=8,
               label="start (centre)"),
        Patch(facecolor=L.DENSITY[2], edgecolor="none", label=r"target $p(x)$"),
        Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=6, alpha=0.4,
               label="final states of all chains"),
        Line2D([], [], color=L.SAMPLE, marker="o", ms=4, lw=1.4,
               label="one example chain"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.93),
               ncol=4, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.5, labelcolor=L.INK, borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    out = f"{args.out_dir}/fig_preconditioning"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
