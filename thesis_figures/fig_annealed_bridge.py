"""Annealing bridge: prior x likelihood -> posterior, down the noise schedule.

The conceptual link between the diffusion chapter and the samplers (Sec. 3.5 ->
4.3). Small-multiples grid (after Blumenthal et al. Fig. 2 / Zhang et al.):

    rows    = noise level sigma, cooling top -> bottom
    columns = diffused prior p_sigma   |   annealed likelihood   |   posterior (their product)

At sigma_max the prior is a blob and the data band is wide, so the posterior is
diffuse; as sigma -> 0 the prior sharpens onto the ring (the data manifold) and the
band tightens, and the posterior collapses onto the TWO points where the band meets
the ring -- the bimodal target the samplers must recover. The two modes are split
along null(A): the geometry, not a hand-placed mixture, makes the posterior bimodal.

Run:  ./.venv/bin/python3 thesis_figures/fig_annealed_bridge.py
"""

from __future__ import annotations

import argparse

import landscape as L
import ring as R

SIGMAS = [1.10, 0.55, 0.22, 0.05]          # cooling, top -> bottom
COLS = [
    (r"prior  $p_{\sigma_t}(x)$", lambda x, s: R.ring_log_prob(x, s)),
    (r"likelihood  $p_{\sigma_t}(y\mid x)$", lambda x, s: R.annealed_loglik(x, s)),
    (r"posterior  $p_{\sigma_t}(x\mid y)$",
     lambda x, s: R.ring_log_prob(x, s) + R.annealed_loglik(x, s)),
]


def panel(ax, fn, sigma, show_truth):
    grid = R.density_grid(lambda x: fn(x, sigma), n=300)
    R.gray_density(ax, grid)
    if show_truth:
        R.draw_truth(ax)
        R.draw_modes(ax)
    R.frame(ax)


def main():
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    L.apply_style()

    nrow, ncol = len(SIGMAS), len(COLS)
    fig, ax = plt.subplots(nrow, ncol, figsize=(7.6, 9.4))

    for i, sigma in enumerate(SIGMAS):
        for j, (_, fn) in enumerate(COLS):
            show = (j == 2)
            panel(ax[i, j], fn, sigma, show)
            if i < nrow - 1:
                ax[i, j].set_xticklabels([])
            if j > 0:
                ax[i, j].set_yticklabels([])
        ax[i, 0].set_ylabel(r"$x_2$  (null)", fontsize=16)
        ax[i, ncol - 1].annotate(rf"$\sigma_t={sigma:.2f}$", xy=(1.04, 0.5),
                                 xycoords="axes fraction", rotation=270,
                                 va="center", ha="left", fontsize=16, color=L.INK)
    for j in range(ncol):
        ax[nrow - 1, j].set_xlabel(r"$x_1$  (measured)", fontsize=16)

    for j, (title, _) in enumerate(COLS):
        ax[0, j].set_title(title, fontsize=17.5, color=L.INK, pad=8)
    from matplotlib.patches import FancyArrowPatch
    fig.add_artist(FancyArrowPatch((0.987, 0.78), (0.987, 0.16),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=12, color=L.SUBTLE, lw=1.0))
    fig.text(0.967, 0.47, r"annealing: $\sigma_t$ decreases", rotation=270,
             va="center", ha="center", fontsize=15, color=L.INK)
    fig.text(0.045, 0.955, "Diffused ring prior meets data band and their "
             r"product collapses to intersections as $\sigma_t\to0$ ",
             fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=9, mew=1.8,
               label=r"true signal $x^\star$"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=L.MODE_EDGE, ms=9,
               label=r"posterior modes $\arg\max_x p(x\mid y)$"),
        Patch(facecolor=L.DENSITY[2], edgecolor="none", label=r"$p_{\sigma_t}$"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.938),
               ncol=3, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.6, labelcolor=L.INK, borderaxespad=0.0)
    fig.tight_layout(rect=[0.02, 0, 0.94, 0.915])
    out = f"{args.out_dir}/fig_annealed_bridge"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
