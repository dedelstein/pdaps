"""How the four samplers are built (Sec. 4.3-4.6): the 2x2 update structure.

All four are annealed estimators: over the schedule the clean estimate x_hat0
sharpens onto the solution x* as the noise level sigma_t cools. They all work
on the benign toy -- the figure shows that they get there by DIFFERENT update
structure, not that one wins (the decoupling benefit is high-dimensional and
does NOT show in 2D; preconditioning is the dimension-independent axis).

Each cell is FOUR artificial outer steps:

    outer loop  : coupled (one reverse-SDE step)  vs  decoupled (denoise -> inner -> re-noise)
    inner step  : isotropic noise (round)          vs  preconditioned noise (stiff-axis first)

  coupled : consecutive estimates are CLOSE -> a smooth local path
  decoupled : each estimate is RE-DERIVED via denoise/re-noise -> consecutive
              estimates can sit far apart (non-local), still converging.

The dashed ring on each step is the noise level sigma_t; it shrinks 1->4, so
re-noise (the step that lowers sigma) is visible as the estimate sharpening.

Run:  ./.venv/bin/python3 thesis_figures/fig_sampler_structure.py
"""

from __future__ import annotations

import numpy as np

import landscape as L

XSTAR = np.array([0.0, 2.0])                 # the solution (true signal)
START = np.array([1.45, 0.15])               # shared rough step-1 estimate
TRAJ = {
    "dps":   [[1.45, 0.15], [1.02, 0.72], [0.55, 1.34], [0.10, 1.88]],   # smooth, diagonal
    "pula":  [[1.45, 0.15], [0.42, 0.55], [0.13, 1.30], [0.00, 1.92]],   # stiff x1 first
    "daps":  [[1.45, 0.15], [-0.75, 1.42], [0.68, 1.66], [0.02, 2.00]],  # non-local
    "pdaps": [[1.45, 0.15], [-0.65, 1.40], [0.22, 2.10], [0.00, 2.00]],  # decoupled jump, then data-consistency solve
}
SIGMA_R = [0.85, 0.52, 0.28, 0.10]           # noise-level ring radius per step
BLUES = ["#9fb2c6", "#6c89ab", "#41648c", L.SAMPLE]

XLIM, YLIM = (-1.9, 2.2), (-0.5, 3.0)


def background(ax):
    """Faint anisotropic posterior at x*: tight on measured x1, loose on null x2."""
    gx = np.linspace(*XLIM, 200)
    gy = np.linspace(*YLIM, 200)
    X, Y = np.meshgrid(gx, gy)
    Z = np.exp(-0.5 * ((X - XSTAR[0]) / 0.45) ** 2 - 0.5 * ((Y - XSTAR[1]) / 1.1) ** 2)
    ax.contourf(X, Y, Z, levels=[0.2, 0.5, 0.8, 1.0], colors=L.DENSITY, zorder=0)


def cell(ax, key, title, subtitle, precond):
    background(ax)
    pts = np.array(TRAJ[key])
    ax.plot([XSTAR[0]], [XSTAR[1]], "x", color=L.TRUTH, ms=13, mew=2.6, zorder=7)
    ax.plot(pts[:, 0], pts[:, 1], "-", color=L.SUBTLE, lw=0.8, alpha=0.6, zorder=3)
    from matplotlib.patches import Circle, Ellipse
    pdaps_r = [0.85, 0.40, 0.16, 0.10]
    pdaps_aniso = [0.0, 0.3, 0.7, 1.0]
    for i, (p, r, c) in enumerate(zip(pts, SIGMA_R, BLUES)):
        if key == "pdaps":
            rr, a = pdaps_r[i], pdaps_aniso[i]
            ax.add_patch(Ellipse(p, 2 * rr * (1 - 0.45 * a), 2 * rr * (1 + 0.35 * a),
                                 fill=False, ec=c, lw=0.8, ls=(0, (2, 2)),
                                 alpha=0.55, zorder=2))
        elif precond:
            ax.add_patch(Ellipse(p, 2 * 0.58 * r, 2 * 1.18 * r, fill=False, ec=c,
                                 lw=0.8, ls=(0, (2, 2)), alpha=0.55, zorder=2))
        else:
            ax.add_patch(Circle(p, r, fill=False, ec=c, lw=0.8, ls=(0, (2, 2)),
                                alpha=0.55, zorder=2))
        ax.plot([p[0]], [p[1]], "o", color=c, ms=8.5, zorder=6)
        ax.text(p[0] + 0.13, p[1] - 0.16, f"{i + 1}", color=c, fontsize=13.5,
                ha="center", va="center", zorder=6)

    ax.set_title(title, fontsize=20, loc="left", pad=3)
    ax.text(0.5, -0.085, subtitle, transform=ax.transAxes, fontsize=14,
            color=L.INK, ha="center", va="top", style="italic")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    import argparse
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()
    L.apply_style()

    fig, ax = plt.subplots(2, 2, figsize=(8.8, 9.2))
    cell(ax[0, 0], "dps", "DPS", "coupled: a smooth local path", precond=False)
    cell(ax[0, 1], "pula", "pULA", "coupled: local, stiff $x_1$ first", precond=True)
    cell(ax[1, 0], "daps", "DAPS", "decoupled: estimate re-derived (non-local)",
         precond=False)
    cell(ax[1, 1], "pdaps", "pDAPS", "decoupled: preconditioned data-consistency",
         precond=True)

    ax[0, 0].annotate("isotropic inner", xy=(0.5, 1.12), xycoords="axes fraction",
                      ha="center", fontsize=18.5, color=L.INK)
    ax[0, 1].annotate("preconditioned inner", xy=(0.5, 1.12), xycoords="axes fraction",
                      ha="center", fontsize=18.5, color=L.INK)
    ax[0, 0].annotate("coupled", xy=(-0.07, 0.5), xycoords="axes fraction",
                      rotation=90, va="center", fontsize=18.5, color=L.INK)
    ax[1, 0].annotate("decoupled", xy=(-0.07, 0.5), xycoords="axes fraction",
                      rotation=90, va="center", fontsize=18.5, color=L.INK)
    fig.text(0.012, 0.97, "Four outer steps each: the estimate sharpens onto "
             r"$x^\star$ as $\sigma_t$ cools (dashed = per-step spread)",
             fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=9, mew=1.8,
               label=r"solution $x^\star$"),
        Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=8,
               label=r"estimate $\hat{x}_0$ (light$\to$dark = step 1$\to$4)"),
        Patch(facecolor=L.DENSITY[2], edgecolor="none", label=r"$p(x\mid y)$"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.948),
               ncol=3, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.6, labelcolor=L.INK, borderaxespad=0.0)
    fig.tight_layout(rect=[0.02, 0, 1, 0.90], h_pad=3.5)
    out = f"{args.out_dir}/fig_sampler_structure"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
