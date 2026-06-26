"""Diffusion / DSM intro figures (Sec. 3.5), on a two-mode prior (same motif as
the sampler figures, centred here since there is no data axis yet). One script,
two separate graphics:

  fig_diffusion        : the forward chain (data -> progressively noised ->
                         ~isotropic Gaussian) and the score-driven reverse chain
                         (noise -> progressively denoised -> data).
  fig_score_tweedie    : the score field nabla log p_sigma (arrows toward the
                         modes) and the one-step Tweedie estimate x_hat0 of a
                         noisy point (Sec. 3.5.1).

Run:  ./.venv/bin/python3 thesis_figures/fig_diffusion.py [--quick]
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

import landscape as L

LIM = 4.0
SIGMAS = [0.0, 0.4, 1.0, 3.5]                         # chain columns (sigma scale)
MODES = torch.tensor([[0.0, 2.0], [0.0, -2.0]])      # centred two-mode prior
STD = 0.5
def score(x, sigma):
    var = STD ** 2 + sigma ** 2
    diff = MODES.unsqueeze(0) - x.unsqueeze(1)       # (B, K, 2)
    w = torch.softmax(-0.5 * (diff ** 2).sum(-1) / var, dim=1)
    return (w.unsqueeze(-1) * diff).sum(1) / var


def tweedie(x, sigma):
    return x + sigma ** 2 * score(x, sigma)


def sample_prior(n):
    idx = torch.randint(0, MODES.shape[0], (n,))
    return MODES[idx] + STD * torch.randn(n, 2)


def forward_snaps(x0, sigmas):
    return [(x0 + s * torch.randn_like(x0)).numpy() for s in sigmas]


def reverse_snaps(sigmas, nb, N, smax):
    """Real reverse VE diffusion; record the population nearest each target sigma."""
    sched = L.make_sigmas(N, smax, 0.02)
    fac = L.factor_steps(sched)
    x = torch.randn(nb, 2) * smax
    targets = sorted([s for s in sigmas if s > 0], reverse=True)
    snaps, ti = {}, 0
    for i in range(N):
        sig = sched[i].item()
        while ti < len(targets) and sig <= targets[ti]:
            snaps[targets[ti]] = x.clone().numpy()
            ti += 1
        x = x + fac[i] * score(x, sig) + np.sqrt(fac[i]) * torch.randn_like(x)
    snaps[0.0] = x.numpy()
    return [snaps[s] for s in sigmas]


def scatter_panel(ax, pts):
    ax.scatter(pts[:, 0], pts[:, 1], s=2.0, c=L.SAMPLE, alpha=0.30,
               linewidths=0, rasterized=True)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_1$", fontsize=13.5, labelpad=1)
    ax.set_ylabel("$x_2$", fontsize=13.5, labelpad=1)


def _square(ax, title):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(MODES[:, 0], MODES[:, 1], s=80, facecolors="none",
               edgecolors=L.MODE_EDGE, linewidths=0.9, zorder=5)
    ax.set_title(title, fontsize=17.5, loc="left")


def draw_score_tweedie(ax):
    """Score field AND Tweedie in one panel: Tweedie is one sigma^2-scaled step
    ALONG the score, so the noisy-point jumps follow the field onto the data."""
    sig = 1.5
    g = np.linspace(-LIM + 0.4, LIM - 0.4, 11)
    X, Y = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
    s = score(pts, sig).numpy()
    u = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
    u = u.reshape(X.shape + (2,))
    ax.quiver(X, Y, u[..., 0], u[..., 1], color=L.SAMPLE, alpha=0.30,
              width=0.004, scale=30, headwidth=4.0, zorder=2)
    noisy = torch.tensor([[-1.8, 2.3], [1.5, 2.5], [0.2, 2.9],
                          [-1.7, -2.3], [1.6, -2.5]])
    x0 = tweedie(noisy, sig)
    noisy, x0 = noisy.numpy(), x0.numpy()
    for i in range(len(noisy)):
        ax.annotate("", xy=x0[i], xytext=noisy[i],
                    arrowprops=dict(arrowstyle="-|>", color=L.TRUTH, lw=1.6,
                                    zorder=6))
    ax.scatter(noisy[:, 0], noisy[:, 1], s=28, c=L.SAMPLE, zorder=7)
    ax.scatter(x0[:, 0], x0[:, 1], marker="x", s=110, c=L.TRUTH, lw=2.0, zorder=8)
    ax.text(0.04, 0.05, r"$\hat{x}_0 = x + \sigma^2\,\nabla\log p_\sigma$"
            "\n(one scaled step along the score)", transform=ax.transAxes,
            fontsize=16, color=L.INK, va="bottom")
    _square(ax, "")   # title removed; narrative + legend carry it


def build_chains(fwd, rev, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ncol = len(SIGMAS)
    fig = plt.figure(figsize=(10.0, 4.8))
    gs = GridSpec(2, ncol, figure=fig, hspace=0.34, wspace=0.12)

    for j, pts in enumerate(fwd):
        ax = fig.add_subplot(gs[0, j])
        scatter_panel(ax, pts)
        ax.set_title(rf"$\sigma={SIGMAS[j]:.1f}$", fontsize=14.5, color=L.INK)
        if j == 0:
            ax.annotate("forward (add noise)", xy=(-0.42, 0.5),
                        xycoords="axes fraction", rotation=90, ha="center",
                        va="center", fontsize=15, color=L.INK)
    for j, pts in enumerate(reversed(rev)):
        ax = fig.add_subplot(gs[1, j])
        scatter_panel(ax, pts)
        ax.set_title(rf"$\sigma={list(reversed(SIGMAS))[j]:.1f}$",
                     fontsize=14.5, color=L.INK)
        if j == 0:
            ax.annotate("reverse (denoise)", xy=(-0.42, 0.5),
                        xycoords="axes fraction", rotation=90, ha="center",
                        va="center", fontsize=15, color=L.INK)
    fig.text(0.012, 0.95, r"Forward reads left$\rightarrow$right (data to noise); "
             r"reverse reads left$\rightarrow$right (noise to data)",
             fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=8,
                      label=r"sample population at noise level $\sigma$")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.935),
               ncol=1, frameon=False, fontsize=13, handletextpad=0.5,
               labelcolor=L.INK, borderaxespad=0.0)
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.11, right=0.99)
    out = f"{out_dir}/fig_diffusion"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")


def build_score_tweedie(out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 6.2))
    draw_score_tweedie(ax)
    fig.text(0.012, 0.95, r"One $\sigma^2$ step along score field maps noisy "
             r"$x$ to posterior mean $\mathbb{E}[x_0\mid x]$",
             fontsize=13.5, color=L.INK)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=L.SAMPLE, lw=1.2, alpha=0.5,
               label=r"score $\nabla\log p_\sigma$"),
        Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=8,
               label=r"noisy point $x$"),
        Line2D([], [], color=L.TRUTH, lw=1.8, label="Tweedie step"),
        Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=9, mew=1.8,
               label=r"estimate $\hat{x}_0$"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=L.MODE_EDGE,
               ms=9, label="data mode"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.925),
               ncol=3, frameon=False, fontsize=12.5, handletextpad=0.5,
               columnspacing=1.4, labelcolor=L.INK, borderaxespad=0.0)
    fig.subplots_adjust(top=0.80, bottom=0.02, left=0.02, right=0.98)
    out = f"{out_dir}/fig_score_tweedie"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()

    L.apply_style()
    torch.manual_seed(0)
    np.random.seed(0)
    nb = 400 if args.quick else 800
    N = 80 if args.quick else 200

    x0 = sample_prior(nb)
    fwd = forward_snaps(x0, SIGMAS)
    rev = reverse_snaps(SIGMAS, nb, N, smax=SIGMAS[-1])

    build_chains(fwd, rev, args.out_dir)
    build_score_tweedie(args.out_dir)


if __name__ == "__main__":
    main()
