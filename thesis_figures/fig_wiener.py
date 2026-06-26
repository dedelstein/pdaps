"""The Wiener process (Sec. 2.5): sqrt(t) spread, roughness, quadratic variation.

Three panels, one simulated process:

  * ensemble  : sample paths of W_t with the +/- sqrt(t) and +/- 2 sqrt(t)
                envelopes (increments W_{t+u} - W_t ~ N(0, u));
  * zooms     : one finely-resolved path magnified x10 and x100 -- equally
                rough at every scale (almost surely nowhere differentiable,
                so dW/dt does not exist);
  * quadratic variation : sum of squared increments vs t for coarse-to-fine
                partitions, converging to the line [W]_t = t -- the identity
                (dW)^2 = dt behind the Ito correction.

Run:  ./.venv/bin/python3 thesis_figures/fig_wiener.py
"""

from __future__ import annotations

import argparse
import numpy as np

import landscape as L

T = 1.0
N_FINE = 2 ** 20          # resolution of the highlighted path
N_ENS = 2 ** 11           # resolution of the background ensemble
N_PATHS = 40
WIN1 = (0.50, 0.60)       # x10 zoom window
WIN2 = (0.50, 0.51)       # x100 zoom window


def brownian(n, rng):
    dw = rng.standard_normal(n) * np.sqrt(T / n)
    w = np.empty(n + 1)
    w[0] = 0.0
    np.cumsum(dw, out=w[1:])
    return np.linspace(0.0, T, n + 1), w


def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()

    L.apply_style()
    rng = np.random.default_rng(7)
    t_fine, w_fine = brownian(N_FINE, rng)

    fig = plt.figure(figsize=(10.6, 3.9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1],
                  width_ratios=[1.3, 0.9, 1.3], hspace=0.45, wspace=0.34)
    ax_ens = fig.add_subplot(gs[:, 0])
    ax_z1 = fig.add_subplot(gs[0, 1])
    ax_z2 = fig.add_subplot(gs[1, 1])
    ax_qv = fig.add_subplot(gs[:, 2])
    for _ in range(N_PATHS):
        tt, ww = brownian(N_ENS, rng)
        ax_ens.plot(tt, ww, color=L.SAMPLE, lw=0.5, alpha=0.18)
    sub = N_FINE // N_ENS
    ax_ens.plot(t_fine[::sub], w_fine[::sub], color=L.SAMPLE, lw=1.1)
    tt = np.linspace(0.0, T, 200)
    for k, ls in ((1, "--"), (2, ":")):
        ax_ens.plot(tt, k * np.sqrt(tt), color=L.TRUTH, lw=0.9, ls=ls)
        ax_ens.plot(tt, -k * np.sqrt(tt), color=L.TRUTH, lw=0.9, ls=ls)
        ax_ens.text(T * 1.01, k * np.sqrt(T), rf"$+{k if k > 1 else ''}\sqrt{{t}}$",
                    fontsize=13.5, color=L.TRUTH, va="center")
    ax_ens.set_xlim(0, T * 1.12)
    ax_ens.set_ylim(-2.6, 2.6)
    ax_ens.set_xticks([0, 0.5, 1])
    ax_ens.set_xlabel(r"time $t$", fontsize=14.5)
    ax_ens.set_ylabel(r"process value  $W_t$", fontsize=15)
    ax_ens.set_title(r"(a)  $W_t\sim\mathcal{N}(0,t)$: spread $\sqrt{t}$",
                     fontsize=14, loc="left")
    for ax, (lo, hi), tag in ((ax_z1, WIN1, r"$\times 10$"),
                              (ax_z2, WIN2, r"$\times 100$")):
        m = (t_fine >= lo) & (t_fine <= hi)
        ax.plot(t_fine[m], w_fine[m], color=L.SAMPLE, lw=0.5)
        ax.set_xlim(lo, hi)
        ax.set_xticks([lo, hi])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=13, length=3)
        ax.text(0.97, 0.92, tag, transform=ax.transAxes, fontsize=14.5,
                color=L.INK, ha="right", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    m1 = (t_fine >= WIN1[0]) & (t_fine <= WIN1[1])
    ax_z1.axvspan(*WIN2, color=L.DENSITY[1], zorder=0)
    ax_ens.axvspan(*WIN1, color=L.DENSITY[1], zorder=0)
    ax_z1.set_title("(b) nowhere differentiable:\n"
                    r"$\frac{dW_t}{dt}$ does not exist", fontsize=14, loc="left")
    n = 2 ** 9                                           # coarse: QV visibly wobbles
    tt = np.linspace(0, T, n + 1)
    wb = w_fine[::N_FINE // n]
    qv_b = np.concatenate([[0.0], np.cumsum(np.diff(wb) ** 2)])
    fs = 0.9 * np.sin(2.5 * np.pi * tt)                  # a smooth path
    qv_s = np.concatenate([[0.0], np.cumsum(np.diff(fs) ** 2)])
    ax_qv.plot([0, T], [0, T], color=L.TRUTH, lw=1.0, ls="--", zorder=2)
    ax_qv.text(0.62, 0.50, r"$[W]_t = t$", fontsize=14.5, color=L.TRUTH,
               rotation=33, rotation_mode="anchor")
    ax_qv.plot(tt, qv_b, color=L.SAMPLE, lw=1.2, zorder=3)
    ax_qv.plot(tt, qv_s, color=L.SUBTLE, lw=1.2, zorder=3)
    ax_qv.text(0.04, 0.93, "a random path,\nbut its quadratic variation\n"
               "is deterministic", fontsize=13.5, color=L.SAMPLE,
               transform=ax_qv.transAxes, va="top")
    ax_qv.text(0.30, 0.10, r"smooth path: $\sum(\Delta f)^2\to 0$",
               fontsize=13.5, color=L.INK)
    ax_qv.set_xlim(0, T * 1.02)
    ax_qv.set_ylim(0, 1.12)
    ax_qv.set_xticks([0, 0.5, 1])
    ax_qv.set_yticks([0, 1])
    ax_qv.set_xlabel(r"time $t$", fontsize=14.5)
    ax_qv.set_ylabel(r"$\sum_i (\Delta x_i)^2$ over $[0,t]$", fontsize=15)
    ax_qv.set_title(r"(c) quadratic variation: $\to t$, not $0$", fontsize=14,
                    loc="left")

    for ax in (ax_ens, ax_qv):
        ax.tick_params(labelsize=13, length=3)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=L.SAMPLE, lw=1.6, label=r"Brownian motion $W_t$"),
        Line2D([], [], color=L.TRUTH, lw=1.4, ls="--",
               label=r"reference ($\pm\sqrt{t}$, $[W]_t=t$)"),
        Line2D([], [], color=L.SUBTLE, lw=1.4, label="a smooth path"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.965),
               ncol=3, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.8, labelcolor=L.INK)
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.05, right=0.99)

    out = f"{args.out_dir}/fig_wiener"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
