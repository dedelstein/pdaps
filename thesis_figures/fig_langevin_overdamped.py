"""Classical vs overdamped Langevin dynamics (Sec. 2.5, the overdamped limit).

Two trajectories in the same quadratic potential f(x) = 1/2 lambda x^2 with
the same diffusion constant D, so they share the stationary distribution
p_inf = N(0, D/lambda) of the diff-eq section:

  * classical   : m dv = -zeta v dt + F(x) dt + sigma dW (Eq. langevin_classical),
                  with sigma = sqrt(2 zeta D) so the position marginal matches.
                  Momentum carries the particle past the minimum (it rings);
  * overdamped  : dx = (1/zeta) F(x) dt + sqrt(2D/zeta) dW
                  (Eq. overdamped_langevin, zeta = 1) -- the inertial term
                  m dv is neglected; guided by the local geometry.

The dashed reference in each panel is the drift-only trajectory (the
stochastic term sigma dW removed). The right margin shows the shared
stationary density (analytic, gray) with the empirical long-run histogram of
each path (blue).

Run:  ./.venv/bin/python3 thesis_figures/fig_langevin_overdamped.py
"""

from __future__ import annotations

import argparse
import numpy as np

import landscape as L

LAM = 1.0          # curvature (same bowl as fig_ode_pde_sde)
DIF = 0.25         # diffusion constant D (same as fig_ode_pde_sde) -> std 0.5
X0 = 2.0
T_SHOW = 20.0      # displayed window
T_EQ = 2000.0      # long run for the empirical stationary histogram
DT = 0.002
M, ZETA_U = 1.0, 0.35      # underdamped: light particle, weak drag
ZETA_O = 1.0               # overdamped drag


def simulate(t_end, seed, overdamped, n_paths=1):
    rng = np.random.default_rng(seed)
    n = int(t_end / DT)
    t = np.linspace(0.0, t_end, n + 1)
    x = np.empty((n + 1, n_paths))
    x[0] = X0
    if overdamped:
        amp = np.sqrt(2 * DIF / ZETA_O * DT)
        for i in range(n):
            x[i + 1] = (x[i] - (LAM / ZETA_O) * x[i] * DT
                        + amp * rng.standard_normal(n_paths))
    else:
        v = np.zeros(n_paths)
        amp = np.sqrt(2 * ZETA_U * DIF * DT) / M
        for i in range(n):
            v += (-(ZETA_U * v + LAM * x[i]) * DT / M
                  + amp * rng.standard_normal(n_paths))
            x[i + 1] = x[i] + v * DT
    return t, x


def deterministic(t, overdamped):
    if overdamped:
        return X0 * np.exp(-(LAM / ZETA_O) * t)
    g = ZETA_U / (2 * M)
    w = np.sqrt(LAM / M - g ** 2)
    return X0 * np.exp(-g * t) * (np.cos(w * t) + g / w * np.sin(w * t))


def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()

    L.apply_style()
    fig = plt.figure(figsize=(10.0, 4.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.14],
                  hspace=0.42, wspace=0.04)

    rows = (
        (False, 11, r"classical   $m\,dv=-\zeta v\,dt+F(x)\,dt+\sigma\,dW_t$",
         "momentum carries the particle\npast the minimum $x^\\star$"),
        (True, 12, r"overdamped ($m\to 0$)   $dx=\frac{1}{\zeta}F(x)\,dt"
         r"+\sqrt{\frac{2D}{\zeta}}\,dW_t$",
         "inertial term $m\\,dv$ neglected: guided by\nthe local geometry, "
         "not momentum"),
    )
    ylim = (-1.3, 2.4)
    xg = np.linspace(*ylim, 200)
    p_inf = np.exp(-0.5 * xg ** 2 / (DIF / LAM))

    for r, (over, seed, eq, note) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        axm = fig.add_subplot(gs[r, 1])

        t, x = simulate(T_SHOW, seed, over)
        ax.plot(t, x[:, 0], color=L.SAMPLE, lw=0.6)
        ax.plot(t, deterministic(t, over), color=L.TRUTH, lw=1.1, ls="--")
        ax.axhline(0.0, color=L.SUBTLE, lw=0.5, ls=":")
        ax.text(0.99, 0.95, r"- - drift only ($\sigma\,dW_t = 0$)", fontsize=13,
                color=L.TRUTH, transform=ax.transAxes, ha="right", va="top")
        ax.set_xlim(0, T_SHOW)
        ax.set_ylim(*ylim)
        ax.set_yticks([-1, 0, 1, 2])
        ax.tick_params(labelsize=13, length=3)
        ax.set_title(eq, fontsize=13.5, loc="left")
        if r == 1:
            ax.set_xlabel(r"time $t$", fontsize=14.5)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(r"position $x$", fontsize=15)
        _, x_long = simulate(T_EQ, seed + 100, over, n_paths=6)
        tail = x_long[int(5 * T_SHOW / DT):].ravel()
        hist, edges = np.histogram(tail, bins=40, range=ylim, density=True)
        axm.fill_betweenx(xg, 0, p_inf / p_inf.max(), color=L.DENSITY[2], lw=0)
        axm.plot(hist / hist.max(), 0.5 * (edges[:-1] + edges[1:]),
                 color=L.SAMPLE, lw=0.9)
        axm.set_xlim(0, 1.25)
        axm.set_ylim(*ylim)
        axm.set_xticks([])
        axm.set_yticks([])
        axm.spines["left"].set_visible(False)
        if r == 0:
            axm.set_title(r"$p_\infty$", fontsize=13.5, loc="left")
        if r == 1:
            axm.text(0.5, -0.07, r"$\mathcal{N}\!\left(0,\frac{D}{\lambda}\right)$",
                     transform=axm.transAxes, fontsize=13.5, color=L.INK,
                     ha="center", va="top")
    fig.text(0.012, 0.96, "The same bowl and diffusion constant $D$: both paths "
             "relax from $x_0$ and fluctuate about the minimum", fontsize=13.5,
             color=L.INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=L.SAMPLE, lw=1.5, label="sample path (and histogram)"),
        Line2D([], [], color=L.TRUTH, lw=1.4, ls="--", label="drift-only solution"),
        Line2D([], [], color=L.SUBTLE, lw=1.2, ls=":", label=r"minimum $x^\star$"),
        Patch(facecolor=L.DENSITY[2], edgecolor="none",
              label=r"stationary $p_\infty$"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.945),
               ncol=4, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.8, labelcolor=L.INK, borderaxespad=0.0)
    fig.subplots_adjust(top=0.78, bottom=0.10, left=0.06, right=0.985)

    out = f"{args.out_dir}/fig_langevin_overdamped"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
