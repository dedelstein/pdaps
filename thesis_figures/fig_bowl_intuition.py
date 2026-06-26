"""Literal sphere-in-a-bowl intuition (Sec. 2.2 opener, before fig_ode_pde_sde).

The physical scene behind the abstract three-panel fig_ode_pde_sde: the SAME
bowl f(x) = 1/2 lambda x^2 (same lambda, D, x0), drawn as a real surface with
a ball on it, three times --

  * ODE : one ball released at x0 rolls deterministically down to the minimum
          x* (fading trail = time);  the force is -grad f along the surface;
  * SDE : the same roll perturbed by noise -- a jittery trail that descends and
          then fluctuates about x*, never settling exactly;
  * PDE : many balls at once; their horizontal positions pile up into the
          stationary density p_inf(x) = N(0, D/lambda) shown along the base.

Deliberately literal (a primer), then fig_ode_pde_sde formalises the same story
as potential + ODE/SDE/PDE panels. Shared visual language: blue = ball/samples,
orange x = the minimum x*, gray tints = density / bowl body.

Run:  ./.venv/bin/python3 thesis_figures/fig_bowl_intuition.py
"""

from __future__ import annotations

import argparse
import numpy as np

import landscape as L

LAM = 1.0          # bowl curvature   (same bowl as fig_ode_pde_sde)
DIF = 0.25         # diffusion constant -> stationary std sqrt(D/lambda) = 0.5
X0 = 2.0           # release position
XLO, XHI = -2.5, 2.5
YB = -0.95         # floor of the marginal-density strip below the bowl


def f(x):
    return 0.5 * LAM * x ** 2


SPHERE_MS = 6.0          # one sphere size everywhere; the alpha fade carries time


def on_curve(ax, x, **kw):
    ax.plot([x], [f(x)], "o", **kw)


def draw_bowl(ax):
    xs = np.linspace(XLO, XHI, 300)
    ax.fill_between(xs, f(xs), f(XHI), color=L.DENSITY[0], lw=0, zorder=0)
    ax.plot(xs, f(xs), color=L.INK, lw=1.4, zorder=2)
    ax.plot([0], [0], "x", color=L.TRUTH, ms=9, mew=1.8, zorder=6)
    ax.text(0.0, -0.28, r"$x^\star$", color=L.TRUTH, fontsize=16,
            ha="center", va="top")
    ax.axhline(0.0, color=L.SUBTLE, lw=0.4, ls=":", zorder=1)


def sde_path(seed, dt=0.004):
    rng = np.random.default_rng(seed)
    n = int(4.0 / dt)
    x = X0
    out = [x]
    for _ in range(n):
        x = x - LAM * x * dt + np.sqrt(2 * DIF * dt) * rng.standard_normal()
        out.append(x)
    return np.array(out)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()

    L.apply_style()
    fig = plt.figure(figsize=(10.0, 3.8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.08)
    ax_ode, ax_sde, ax_pde = (fig.add_subplot(gs[0, j]) for j in range(3))
    draw_bowl(ax_ode)
    t = np.linspace(0.0, 3.0, 8)
    xr = X0 * np.exp(-LAM * t)
    ax_ode.plot(xr, f(xr), "-", color=L.SAMPLE, lw=0.7, alpha=0.4, zorder=3)
    for k, x in enumerate(xr):
        a = 0.25 + 0.75 * k / (len(xr) - 1)
        on_curve(ax_ode, x, color=L.SAMPLE, ms=SPHERE_MS, alpha=a, mec="none",
                 zorder=4)
    on_curve(ax_ode, X0, mfc="white", mec=L.SAMPLE, ms=SPHERE_MS, mew=1.3, zorder=5)
    tang = np.array([1.0, LAM * X0]) / np.hypot(1.0, LAM * X0)
    base = np.array([X0, f(X0)])
    tip = base - 0.7 * tang
    ax_ode.annotate("", xy=tip, xytext=base,
                    arrowprops=dict(arrowstyle="-|>", color=L.SUBTLE, lw=1.1))
    ax_ode.text(X0 - 0.30, f(X0) + 0.18, r"$-\nabla f$", color=L.SUBTLE,
                fontsize=14.5, ha="right")
    ax_ode.text(X0 + 0.05, f(X0) + 0.12, r"$x_0$", color=L.SAMPLE, fontsize=14.5)
    ax_ode.set_title("ODE", fontsize=16,
                     loc="left")
    ax_ode.set_ylabel(r"potential energy  $f(x)=\frac{1}{2}\lambda x^2$",
                      fontsize=15)
    draw_bowl(ax_sde)
    path = sde_path(seed=4)
    sub = path[::len(path) // 60]
    ax_sde.plot(sub, f(sub), "-", color=L.SAMPLE, lw=0.5, alpha=0.5, zorder=3)
    for k, x in enumerate(sub):
        a = 0.18 + 0.55 * k / (len(sub) - 1)
        on_curve(ax_sde, x, color=L.SAMPLE, ms=SPHERE_MS, alpha=a, mec="none",
                 zorder=4)
    on_curve(ax_sde, path[-1], color=L.SAMPLE, ms=SPHERE_MS, zorder=5)
    on_curve(ax_sde, X0, mfc="white", mec=L.SAMPLE, ms=SPHERE_MS, mew=1.3, zorder=5)
    ax_sde.text(0.5, 0.86, "descends, then fluctuates\nabout $x^\\star$ forever",
                transform=ax_sde.transAxes, fontsize=13, color=L.SUBTLE,
                ha="center", va="top")
    ax_sde.set_title("SDE", fontsize=16,
                     loc="left")
    draw_bowl(ax_pde)
    rng = np.random.default_rng(1)
    std = np.sqrt(DIF / LAM)
    xs_balls = rng.normal(0.0, std, 44)
    xs_balls = xs_balls[(xs_balls > XLO) & (xs_balls < XHI)]
    ax_pde.scatter(xs_balls, f(xs_balls), s=SPHERE_MS ** 2, c=L.SAMPLE, alpha=0.45,
                   linewidths=0, zorder=4)
    xg = np.linspace(XLO, XHI, 300)
    p = np.exp(-0.5 * xg ** 2 / (DIF / LAM))
    h = p / p.max() * (-(YB + 0.1))
    ax_pde.fill_between(xg, -0.1, -0.1 + h, color=L.DENSITY[2], lw=0, zorder=1)
    ax_pde.plot(xg, -0.1 + h, color=L.SUBTLE, lw=0.7, zorder=2)
    ax_pde.text(0.0, YB + 0.06, r"$p_\infty(x)=\mathcal{N}\!\left(0,\frac{D}{\lambda}\right)$",
                color=L.INK, fontsize=13.5, ha="center", va="bottom")
    ax_pde.text(0.5, 0.86, "many spheres at once:\ntheir positions are the density",
                transform=ax_pde.transAxes, fontsize=13, color=L.SUBTLE,
                ha="center", va="top")
    ax_pde.set_title("PDE", fontsize=16,
                     loc="left")

    for ax in (ax_ode, ax_sde, ax_pde):
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(YB, 3.25)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([0, 1, 2, 3])
        ax.tick_params(labelsize=13, length=3)
        ax.set_xlabel(r"position $x$", fontsize=15)
        ax.spines["bottom"].set_visible(False)
    for ax in (ax_sde, ax_pde):
        ax.set_yticklabels([])
    fig.text(0.012, 0.95, "Deterministic descent (ODE), the same descent under "
             "noise (SDE), and the stationary ensemble (PDE)",
             fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=8, label="sphere"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=L.SAMPLE, ms=8,
               label=r"release point $x_0$"),
        Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=9, mew=1.8,
               label=r"minimum $x^\star$"),
        Patch(facecolor=L.DENSITY[2], edgecolor="none", label=r"$p_\infty(x)$"),
        Line2D([], [], color=L.SAMPLE, lw=1.4, alpha=0.5, label="trail (= time)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.935),
               ncol=5, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.4, labelcolor=L.INK, borderaxespad=0.0)
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.07, right=0.99)

    out = f"{args.out_dir}/fig_bowl_intuition"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
