"""k-space acquisition and undersampling (Sec. on k-space -> fastMRI masking).

Left: the Cartesian acquisition trajectory. During each repetition (TR) the
readout coordinate K1(t) = gamma G1 t advances continuously along one line,
while the phase-encode coordinate K2 advances once per repetition -- so
acquisition time is proportional to the number of phase-encode lines.

Right: the undersampling operator P as line masks (acquired lines in ink):
fully sampled (Nyquist), equispaced R=4, and random R=4, the latter two
retaining the fully-sampled ACS block at the k-space origin.

Run:  ./.venv/bin/python3 thesis_figures/fig_kspace_sampling.py
"""

from __future__ import annotations

import argparse
import numpy as np

import landscape as L

N_LINES = 96          # phase-encode lines in the mask panels
R = 4                 # acceleration factor
N_ACS = 12            # fully-sampled centre block


def masks():
    full = np.ones(N_LINES, bool)
    acs = np.zeros(N_LINES, bool)
    acs[N_LINES // 2 - N_ACS // 2: N_LINES // 2 + N_ACS // 2] = True
    equi = acs.copy()
    equi[::R] = True
    rng = np.random.default_rng(5)
    rand = acs.copy()
    budget = equi.sum() - acs.sum()
    pool = np.flatnonzero(~acs)
    rand[rng.choice(pool, budget, replace=False)] = True
    return [("fully sampled (Nyquist)", full),
            (rf"equispaced $R={R}$ + ACS", equi),
            (rf"random $R={R}$ + ACS", rand)]


def draw_trajectory(ax):
    rows = np.linspace(-0.40, 0.40, 9)
    n_done = 4
    for j, k2 in enumerate(rows):
        if j < n_done:
            ax.annotate("", xy=(0.48, k2), xytext=(-0.48, k2),
                        arrowprops=dict(arrowstyle="-|>", color=L.SAMPLE, lw=1.2))
        else:
            ax.plot([-0.48, 0.48], [k2, k2], color=L.DENSITY[2], lw=1.0)
        if j < n_done - 1:
            ax.annotate("", xy=(-0.54, rows[j + 1]), xytext=(-0.54, k2),
                        arrowprops=dict(arrowstyle="-|>", color=L.TRUTH, lw=1.0))
    for j in range(n_done):
        ax.text(0.51, rows[j], rf"$TR_{{{j + 1}}}$", fontsize=13, color=L.INK,
                va="center")
    ax.text(0.0, rows[n_done - 1] + 0.045,
            r"readout: $K_1(t)=\gamma G_1 t$, continuous",
            fontsize=13.5, color=L.SAMPLE, ha="center")
    ax.text(-0.585, np.mean(rows[:n_done]),
            "phase encode $K_2$:\none step\nper repetition",
            fontsize=13.5, color=L.TRUTH, ha="right", va="center")
    ax.text(0.0, rows[-1] + 0.06, "not yet acquired", fontsize=13, color=L.INK,
            ha="center")
    ax.set_xlim(-0.88, 0.62)
    ax.set_ylim(-0.50, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.annotate("", xy=(0.18, -0.47), xytext=(-0.18, -0.47),
                arrowprops=dict(arrowstyle="-|>", color=L.INK, lw=0.6))
    ax.text(0.0, -0.50, r"$K_1$ (readout)", fontsize=14.5, color=L.INK,
            ha="center", va="top")
    ax.set_title("Cartesian acquisition", fontsize=14.5, loc="left")


def draw_mask(ax, title, mask):
    rows = np.zeros(4 * N_LINES)
    for i in np.flatnonzero(mask):
        rows[4 * i: 4 * i + 3] = 1.0
    img = np.repeat(rows[:, None], N_LINES, axis=1)
    ax.imshow(img, cmap="gray_r", aspect="auto", interpolation="nearest",
              vmin=0, vmax=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14.5, loc="left")
    ax.text(0.02, -0.06, f"{int(mask.sum())}/{N_LINES} lines",
            transform=ax.transAxes, fontsize=13, color=L.INK, va="top")


def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()

    L.apply_style()
    fig = plt.figure(figsize=(10.5, 3.7))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.9, 1, 1, 1], wspace=0.30)

    draw_trajectory(fig.add_subplot(gs[0, 0]))
    axes = [fig.add_subplot(gs[0, k]) for k in (1, 2, 3)]
    for ax, (title, mask) in zip(axes, masks()):
        draw_mask(ax, title, mask)
    axes[0].set_ylabel(r"phase-encode line  $K_2$", fontsize=14.5)
    lo = 4 * (N_LINES // 2 - N_ACS // 2)
    hi = lo + 4 * N_ACS
    axes[2].text(N_LINES * 1.03, (lo + hi) / 2, "ACS", fontsize=13.5,
                 color=L.TRUTH, ha="left", va="center", clip_on=False)
    for k in (1, 2):
        for y in (lo - 0.5, hi - 0.5):
            axes[k].axhline(y, color=L.TRUTH, lw=0.7)
    fig.text(0.012, 0.935, "Acquisition time is proportional to the number of "
             "phase-encode lines;", fontsize=13.5, color=L.INK)
    fig.text(0.012, 0.875, "the masks drop lines (acquired = ink) while retaining "
             "the fully-sampled ACS block at the origin", fontsize=13.5, color=L.INK)
    fig.subplots_adjust(top=0.74, bottom=0.10, left=0.03, right=0.99)

    out = f"{args.out_dir}/fig_kspace_sampling"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
