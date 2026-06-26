"""VE vs VP forward processes (diffusion section, after Eqs. ve_sde / vp_sde).

The two forward-noising strategies on the same initial signal x0, drawn in the
visual language of the background chapter's fig_ode_pde_sde (sample paths,
mean +/- 2 sigma envelope, stationary margin):

  * VE  : dx = sqrt(d sigma_t^2/dt) dW -- zero drift, the signal coordinates
          remain stationary while the variance explodes;
  * VP  : dx = -(1/2) beta x dt + sqrt(beta) dW -- the restorative drift
          contracts the signal toward the origin while the variance stays
          bounded; this is the OU process of the background chapter
          (Eq. sde_eq) with time-dependent drift.

Run:  ./.venv/bin/python3 thesis_figures/fig_ve_vp.py
"""

from __future__ import annotations

import argparse
import numpy as np

import landscape as L

X0 = 2.0
T = 1.0
SIGMA_MAX = 2.0       # VE noise scale at t = T
BETA = 6.0            # VP rate (constant beta(t) for legibility)
YLO, YHI = -3.2, 6.2
N_PATHS = 4
DT = 0.001


def ve_sigma(t):
    return SIGMA_MAX * t


def vp_mu(t):
    return X0 * np.exp(-0.5 * BETA * t)


def vp_var(t):
    return 1.0 - np.exp(-BETA * t)


def paths(overdrift, seed):
    rng = np.random.default_rng(seed)
    n = int(T / DT)
    t = np.linspace(0.0, T, n + 1)
    x = np.full((n + 1, N_PATHS), X0)
    for i in range(n):
        if overdrift:                       # VP
            x[i + 1] = (x[i] - 0.5 * BETA * x[i] * DT
                        + np.sqrt(BETA * DT) * rng.standard_normal(N_PATHS))
        else:                               # VE, sigma_t = SIGMA_MAX * t
            dvar = ve_sigma(t[i + 1]) ** 2 - ve_sigma(t[i]) ** 2
            x[i + 1] = x[i] + np.sqrt(dvar) * rng.standard_normal(N_PATHS)
    return t, x


def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()

    L.apply_style()
    fig = plt.figure(figsize=(10.0, 3.7))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 0.14, 0.10, 1, 0.14],
                  wspace=0.08)
    tg = np.linspace(0.0, T, 300)
    xg = np.linspace(YLO, YHI, 300)

    panels = (
        (0, False, r"VE   $d\mathbf{x}=\sqrt{\frac{d\sigma_t^2}{dt}}\,d\mathbf{W}_t$",
         np.full_like(tg, X0), ve_sigma(tg),
         "zero drift: signal coordinates stationary,\nvariance explodes",
         np.exp(-0.5 * (xg - X0) ** 2 / ve_sigma(T) ** 2),
         r"$\mathcal{N}(x_0,\sigma_{\max}^2)$"),
        (3, True, r"VP   $d\mathbf{x}=-\frac{1}{2}\beta\mathbf{x}\,dt"
         r"+\sqrt{\beta}\,d\mathbf{W}_t$",
         vp_mu(tg), np.sqrt(vp_var(tg)),
         "restorative drift contracts the signal,\nvariance bounded",
         np.exp(-0.5 * xg ** 2),
         r"$\mathcal{N}(0,1)$"),
    )

    for col, over, eq, mu, sig, note, p_end, p_lbl in panels:
        ax = fig.add_subplot(gs[0, col])
        axm = fig.add_subplot(gs[0, col + 1])
        t, x = paths(over, seed=8 + col)
        for k in range(N_PATHS):
            ax.plot(t, x[:, k], color=L.SAMPLE, lw=0.55, alpha=0.65)
        ax.plot(tg, mu, color=L.TRUTH, lw=1.2, ls="--")
        for sgn in (+2, -2):
            ax.plot(tg, mu + sgn * sig, color=L.SUBTLE, lw=0.7, ls=":")
        ax.text(0.03, 0.94, note, fontsize=11.5, color=L.INK,
                transform=ax.transAxes, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
        ax.text(0.70, 0.06, r"$\mu_t\pm 2\sigma_t$", fontsize=13.5,
                color=L.INK, transform=ax.transAxes)
        ax.set_xlim(0, T)
        ax.set_ylim(YLO, YHI)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "$T$"])
        ax.set_xlabel(r"diffusion time $t$", fontsize=14.5)
        ax.tick_params(labelsize=13, length=3)
        ax.set_title(eq, fontsize=13, loc="left")
        if col == 0:
            ax.set_yticks([-2, 0, 2, 4, 6])
            ax.set_ylabel(r"signal $x$", fontsize=15)
        else:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)

        axm.fill_betweenx(xg, 0, p_end / p_end.max(), color=L.DENSITY[2], lw=0)
        axm.plot(p_end / p_end.max(), xg, color=L.SUBTLE, lw=0.7)
        axm.set_xlim(0, 1.3)
        axm.set_ylim(YLO, YHI)
        axm.set_xticks([])
        axm.set_yticks([])
        axm.spines["left"].set_visible(False)
        axm.set_title(r"$p_T$", fontsize=13.5, loc="left")
        axm.text(0.10, 0.04, p_lbl, transform=axm.transAxes, fontsize=13,
                 color=L.INK)
    fig.text(0.012, 0.95, "The same $x_0$ under the variance-exploding (VE) and "
             "variance-preserving (VP) forward SDEs", fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=L.SAMPLE, lw=1.5, label="sample paths"),
        Line2D([], [], color=L.TRUTH, lw=1.4, ls="--", label=r"mean $\mu_t$"),
        Line2D([], [], color=L.SUBTLE, lw=1.2, ls=":", label=r"$\mu_t\pm 2\sigma_t$"),
        Patch(facecolor=L.DENSITY[2], edgecolor="none",
              label=r"$p_T$ (prior at $t=T$)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.935),
               ncol=4, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.8, labelcolor=L.INK, borderaxespad=0.0)
    fig.subplots_adjust(top=0.772, bottom=0.15, left=0.05, right=0.99)

    out = f"{args.out_dir}/fig_ve_vp"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
