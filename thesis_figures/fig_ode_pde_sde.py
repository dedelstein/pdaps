"""One system, three levels of description (Sec. 2.2, the diff-eq progression).

The quadratic-bowl running example of the differential-equations section,
drawn as four panels sharing the position axis:

  * potential : f(x) = 1/2 lambda x^2 drawn sideways, the sphere at x0;
  * ODE       : dx/dt = -lambda x -- one deterministic descent per x0 (the flow);
  * SDE       : dx = -lambda x dt + sqrt(2D) dW -- the same descent, perturbed;
  * PDE       : Fokker-Planck -- the density of all such paths,
                N(mu(t), sigma^2(t)), with a second initial condition merging
                into the same p_inf (diffusion destroys information, so the
                operator T_{-t} is undefined);
  * margin    : the shared stationary density p_inf = N(0, D/lambda).

Run:  ./.venv/bin/python3 thesis_figures/fig_ode_pde_sde.py
"""

from __future__ import annotations

import argparse
import numpy as np

import landscape as L

LAM = 1.0          # curvature of the bowl
DIF = 0.25         # diffusion constant -> stationary std sqrt(D/lambda) = 0.5
X0 = 2.0           # the sphere's initial position
X0_ALT = -1.3      # a second initial condition (for the PDE merging story)
T = 4.0
XLO, XHI = -1.6, 2.6


def mu(t, x0=X0):
    return x0 * np.exp(-LAM * t)


def var(t):
    return DIF / LAM * (1.0 - np.exp(-2 * LAM * t))


def sde_paths(n, dt=0.002, seed=3):
    rng = np.random.default_rng(seed)
    steps = int(T / dt)
    t = np.linspace(0.0, T, steps + 1)
    x = np.full(n, X0)
    out = np.empty((steps + 1, n))
    out[0] = x
    for i in range(steps):
        x = x - LAM * x * dt + np.sqrt(2 * DIF * dt) * rng.standard_normal(n)
        out[i + 1] = x
    return t, out


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()

    L.apply_style()
    fig = plt.figure(figsize=(11.7, 4.7))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[0.40, 1.0, 1.0, 1.25, 0.16],
                  wspace=0.40)
    ax_pot = fig.add_subplot(gs[0, 0])
    ax_ode = fig.add_subplot(gs[0, 1])
    ax_sde = fig.add_subplot(gs[0, 2])
    ax_pde = fig.add_subplot(gs[0, 3])
    ax_inf = fig.add_subplot(gs[0, 4])
    xs = np.linspace(XLO, XHI, 300)
    ax_pot.plot(0.5 * LAM * xs ** 2, xs, color=L.INK, lw=1.0)
    ax_pot.plot([0.5 * LAM * X0 ** 2], [X0], "o", color=L.SAMPLE, ms=7, zorder=5)
    ax_pot.plot([0.0], [0.0], marker="x", color=L.TRUTH, ms=9, mew=1.4, zorder=5)
    ax_pot.annotate("", xy=(0.5 * LAM * 1.05 ** 2, 1.05),
                    xytext=(0.5 * LAM * 1.55 ** 2, 1.55),
                    arrowprops=dict(arrowstyle="-|>", color=L.SUBTLE, lw=1.0))
    ax_pot.text(0.5, -0.07, r"$f(x)=\frac{1}{2}\lambda x^2$",
                transform=ax_pot.transAxes, fontsize=15, color=L.INK,
                ha="center", va="top")
    ax_pot.text(0.30, X0 + 0.16, r"$x_0$", fontsize=14.5, color=L.SAMPLE)
    ax_pot.text(0.55, -0.05, r"$x^\star$", fontsize=14.5, color=L.TRUTH, va="top")
    ax_pot.set_xlim(0, 3.7)
    ax_pot.set_xticks([])
    ax_pot.set_yticks([-1, 0, 1, 2])
    ax_pot.set_ylabel(r"position $x$", fontsize=15)
    ax_pot.set_title("potential", fontsize=12, loc="left")
    t = np.linspace(0.0, T, 300)
    for x0 in (1.2, 0.4, -0.6, X0_ALT):
        ax_ode.plot(t, mu(t, x0), color=L.SUBTLE, lw=0.7, alpha=0.8)
    ax_ode.plot(t, mu(t), color=L.SAMPLE, lw=1.6)
    ax_ode.text(0.46, 0.82, r"$x(t)=x_0 e^{-\lambda t}$",
                transform=ax_ode.transAxes, fontsize=13, color=L.SAMPLE)
    ax_ode.text(0.46, 0.75, "one path per $x_0$",
                transform=ax_ode.transAxes, fontsize=10.5, color=L.INK)
    ax_ode.set_title(r"ODE   $\frac{dx}{dt}=-\lambda x$", fontsize=12, loc="left")
    ts, paths = sde_paths(4)
    for k in range(paths.shape[1]):
        ax_sde.plot(ts, paths[:, k], color=L.SAMPLE, lw=0.6, alpha=0.65)
    ax_sde.plot(t, mu(t), color=L.TRUTH, lw=1.3, ls="--")
    ax_sde.text(0.40, 0.82, "- - ODE trajectory", fontsize=11, color=L.TRUTH,
                transform=ax_sde.transAxes)
    ax_sde.set_title(r"SDE   $dx=-\lambda x\,dt+\sqrt{2D}\,dW_t$",
                     fontsize=12, loc="left")
    tg = np.linspace(1e-3, T, 400)
    xg = np.linspace(XLO, XHI, 320)
    sig = np.sqrt(var(tg))
    P = np.exp(-0.5 * (xg[:, None] - mu(tg)[None, :]) ** 2 / sig[None, :] ** 2)
    cmap = LinearSegmentedColormap.from_list("ink", ["#ffffff", "#aaaaaa"])
    ax_pde.pcolormesh(tg, xg, P, cmap=cmap, shading="auto", rasterized=True)
    ax_pde.plot(tg, mu(tg), color=L.SAMPLE, lw=1.3)
    for sgn in (+2, -2):
        ax_pde.plot(tg, mu(tg) + sgn * sig, color=L.SUBTLE, lw=0.7, ls=":")
    ax_pde.plot(tg, mu(tg, X0_ALT), color=L.SAMPLE, lw=1.1, ls="--")
    ax_pde.text(0.34, 0.82, r"$\mu_t\pm 2\sigma(t)$", fontsize=12,
                color=L.SAMPLE, transform=ax_pde.transAxes)
    ax_pde.text(0.04, 0.02, "a different $p(x,0)$", fontsize=10.5, color=L.SAMPLE,
                transform=ax_pde.transAxes)
    ax_pde.text(0.97, 0.03, r"both reach $p_\infty$",
                fontsize=10.5, color=L.INK, ha="right", va="bottom",
                transform=ax_pde.transAxes)
    ax_pde.set_title(r"PDE   $\partial_t p=\lambda\,\partial_x(x p)+D\,\partial_x^2 p$",
                     fontsize=12, loc="left")
    p_inf = np.exp(-0.5 * xg ** 2 / (DIF / LAM))
    ax_inf.fill_betweenx(xg, 0, p_inf, color=L.DENSITY[2], lw=0)
    ax_inf.plot(p_inf, xg, color=L.SUBTLE, lw=0.7)
    ax_inf.set_xticks([])
    ax_inf.set_xlim(0, 1.35)
    ax_inf.spines["left"].set_visible(False)
    ax_inf.set_title(r"$p_\infty$", fontsize=13.5, loc="left")
    ax_inf.text(0.08, 0.84, r"$\mathcal{N}\!\left(0,\frac{D}{\lambda}\right)$",
                transform=ax_inf.transAxes, fontsize=14.5, color=L.INK)

    for ax in (ax_pot, ax_ode, ax_sde, ax_pde, ax_inf):
        ax.set_ylim(XLO, XHI)
    for ax in (ax_ode, ax_sde, ax_pde):
        ax.set_xlim(0, T)
        ax.set_xticks([0, 2, 4])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=13, length=3)
        ax.set_xlabel(r"time $t$", fontsize=14.5)
    ax_pot.tick_params(labelsize=13, length=3)
    fig.text(0.012, 0.955, "Sphere in the bowl: deterministic descent (ODE), "
             "the same descent perturbed by noise (SDE), and the ensemble of all "
             "such paths (PDE)", fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=9, mew=1.8,
               label=r"minimum $x^\star$"),
        Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=8,
               label=r"sphere at $x_0$"),
        Line2D([], [], color=L.SAMPLE, lw=1.8, label="a trajectory"),
        Line2D([], [], color=L.TRUTH, lw=1.6, ls="--", label="ODE trajectory"),
        Patch(facecolor=L.DENSITY[2], edgecolor="none",
              label=r"density $p(x,t)$"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.945),
               ncol=5, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.6, labelcolor=L.INK)
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.05, right=0.99)

    out = f"{args.out_dir}/fig_ode_pde_sde"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
