"""Gradient-descent intro figures (Sec. 2.2, recalled at Sec. 4.1).

Two bespoke single-quadratic illustrations in the shared visual language:

  A1 conditioning : a round well (kappa=1, straight to the minimum in a few steps)
                    vs a stretched valley (kappa>>1, slow zigzag) -- motivates the
                    condition number and the step-count bound k ~ (kappa/2) log(1/eps).
  A2 null axis    : a rank-deficient quadratic whose minimiser is a whole LINE
                    x* + null(A): descent fixes the measured axis and leaves the
                    unmeasured axis untouched. Foreshadows null(A).

Run:  ./.venv/bin/python3 thesis_figures/fig_gradient_descent.py
"""

from __future__ import annotations

import numpy as np

import landscape as L


def gd_path(grad, eta, start, steps):
    x = np.asarray(start, float)
    pts = [x.copy()]
    for _ in range(steps):
        x = x - eta * grad(x)
        pts.append(x.copy())
    return np.array(pts)


def energy_contours(ax, f, lim, n=240, levels=10):
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    lv = np.geomspace(Z.max() * 1e-3 + 1e-9, Z.max(), levels)
    ax.contour(X, Y, Z, levels=lv, colors=L.MODE_EDGE, linewidths=0.4, alpha=0.7)


def draw_path(ax, path):
    ax.plot(path[:, 0], path[:, 1], "-", color=L.SAMPLE, lw=1.0, alpha=0.9)
    ax.plot(path[:, 0], path[:, 1], ".", color=L.SAMPLE, ms=3.0)
    ax.plot([path[0, 0]], [path[0, 1]], "o", mfc="none", mec=L.SAMPLE, ms=7)


def minimum_marker(ax, xy=(0, 0)):
    ax.plot([xy[0]], [xy[1]], "x", color=L.TRUTH, ms=12, mew=1.6)


def cg_path(A_mat, start):
    """CG minimising 1/2 x^T A x (b = 0); reaches x* = 0 exactly in n=2 steps."""
    x = np.asarray(start, float)
    pts = [x.copy()]
    r = -A_mat @ x
    p = r.copy()
    for _ in range(2):
        Ap = A_mat @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        pts.append(x.copy())
        r_new = r - alpha * Ap
        p = r_new + (r_new @ r_new) / (r @ r) * p
        r = r_new
    return np.array(pts)


def fig_conditioning(out):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)
    lim = 4.0
    a = 20.0                                            # ill-conditioned: kappa = 20
    energy_contours(ax1, lambda X, Y: 0.5 * (X ** 2 + Y ** 2), lim)
    p1 = gd_path(lambda x: x, eta=0.55, start=(3.2, 3.0), steps=14)
    draw_path(ax1, p1)
    minimum_marker(ax1)
    ax1.set_title(r"GD,  $\kappa = 1$", fontsize=16, loc="left")
    ax1.text(0.04, 0.05, f"{len(p1)-1} steps", transform=ax1.transAxes,
             fontsize=15, color=L.INK, va="bottom")
    energy_contours(ax2, lambda X, Y: 0.5 * (a * X ** 2 + Y ** 2), lim)
    p2 = gd_path(lambda x: np.array([a * x[0], x[1]]),
                 eta=1.9 / a, start=(2.6, 3.4), steps=70)
    draw_path(ax2, p2)
    minimum_marker(ax2)
    ax2.set_title(r"GD,  $\kappa = 20$", fontsize=16, loc="left")
    ax2.text(0.04, 0.05, f"{len(p2)-1} steps\n" r"$\mathcal{O}(\kappa)$",
             transform=ax2.transAxes, fontsize=15, color=L.INK, va="bottom")
    energy_contours(ax3, lambda X, Y: 0.5 * (a * X ** 2 + Y ** 2), lim)
    cg = cg_path(np.diag([a, 1.0]), np.array([2.6, 3.4]))
    draw_path(ax3, cg)
    minimum_marker(ax3)
    ax3.set_title(r"CG,  $\kappa = 20$", fontsize=16, loc="left")
    for i in (1, 2):
        ax3.annotate(f"step {i}", xy=cg[i] + np.array([0.15, 0.15]),
                     fontsize=12, color=L.SAMPLE)
    ax3.text(0.04, 0.05, "2 steps (exact)\n" r"$\mathcal{O}(\sqrt{\kappa})$",
             transform=ax3.transAxes, fontsize=15, color=L.INK, va="bottom")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        ax.tick_params(labelsize=13, length=3)
        ax.set_xlabel("$x_1$", fontsize=16)
    ax1.set_ylabel("$x_2$", fontsize=16)
    fig.text(0.012, 0.95, r"GD's step count grows with the condition number "
             r"$\kappa$; conjugate directions reach $x^\star$ in $n$ steps",
             fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="none", mec=L.SAMPLE, ms=8,
               label=r"start $x_0$"),
        Line2D([], [], color=L.SAMPLE, marker="o", ms=5, lw=1.4,
               label=r"iterates $x_k$ (line = path)"),
        Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=9, mew=1.8,
               label=r"minimiser $x^\star$"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.935),
               ncol=3, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.8, labelcolor=L.INK, borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")


def fig_nullspace(out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    lim = 4.0
    a = 10.0
    g = np.linspace(-lim, lim, 260)
    X, Y = np.meshgrid(g, g)
    Z = 0.5 * a * X ** 2
    ax.contourf(X, Y, Z, levels=np.linspace(0, Z.max(), 7),
                colors=["#f4f4f4", "#e8e8e8", "#dcdcdc", "#d0d0d0",
                        "#c4c4c4", "#b8b8b8"], antialiased=True)
    ax.axvline(0.0, color=L.TRUTH, lw=2.4, alpha=0.95, zorder=1)
    starts = [(-3.2, 2.4), (3.0, 0.7), (-2.8, -1.1), (3.3, -2.6)]
    for s in starts:
        p = gd_path(lambda x: np.array([a * x[0], 0.0]), eta=0.035,
                    start=s, steps=16)
        draw_path(ax, p)
        ax.annotate("", xy=(p[-1, 0], p[-1, 1]), xytext=(p[-3, 0], p[-3, 1]),
                    arrowprops=dict(arrowstyle="->", color=L.SAMPLE, lw=1.2))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])
    ax.tick_params(labelsize=13, length=3)
    ax.set_xlabel("measured axis  $x_1$  (steep: data constrains it)", fontsize=16)
    ax.set_ylabel(r"unmeasured axis  $x_2 \in \mathrm{null}(A)$  (flat)", fontsize=16)
    fig.text(0.012, 0.95, "Each descent travels to the valley floor and stops. Unmeasured axis remains unchanged.",
             fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="none", mec=L.SAMPLE, ms=8,
               label=r"start $x_0$"),
        Line2D([], [], color=L.SAMPLE, marker="o", ms=5, lw=1.4,
               label=r"iterates $x_k$ (arrow = direction)"),
        Line2D([], [], color=L.TRUTH, lw=2.4,
               label=r"minimisers $x^\star + \mathrm{null}(A)$"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.935),
               ncol=3, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.8, labelcolor=L.INK, borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()
    L.apply_style()
    fig_conditioning(f"{args.out_dir}/fig_gd_conditioning")
    fig_nullspace(f"{args.out_dir}/fig_gd_nullspace")


if __name__ == "__main__":
    main()
