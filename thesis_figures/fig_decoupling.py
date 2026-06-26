"""Decoupling figure (coupled vs decoupled OUTER loop, Sec. 4.3-4.4).

Companion to fig_preconditioning: that figure isolates the preconditioning COLUMN
of the 2x2 (iso vs preconditioned inner step, on a stiff Gaussian where DPS==DAPS);
this isolates the decoupling ROW (coupled vs decoupled outer loop, both
NON-preconditioned). Together they span the 2x2 that fig_sampler_structure
schematises.

Setup is the DAPS paper's own 2-D toy (their Fig. 4), NOT the stiff null-space
landscape: a FULL-RANK, well-posed problem (no null axis, so the inner Langevin
does not diverge) with a two-mode prior and an AMBIGUOUS measurement placed
between the modes, so the posterior stays bimodal and bounded. Both lobes are
valid; there is no single "truth".

The honest, visible difference is the one the DAPS paper actually claims -- the
TRAJECTORY character, not a coverage contest:

  * coupled (DPS): one reverse-SDE step per level -> consecutive iterates are
    CLOSE; the chain walks a smooth, local path and settles into one lobe.
  * decoupled (DAPS): denoise -> inner Langevin -> re-noise -> consecutive
    iterates can be FAR apart; the chain makes non-local jumps and can hop lobes.

DAPS-Fig-4 style: posterior density + a couple of example chains coloured by noise
scale (light = hot/early, dark blue = cold/late). No truth marker.

Run:  ./.venv/bin/python3 thesis_figures/fig_decoupling.py
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import landscape as L

torch.set_default_dtype(torch.float64)
MODES = torch.tensor([[-1.5, 0.0], [1.5, 0.0]])
LOG_W = torch.log(torch.full((2,), 0.5))
PRIOR_VAR = 0.16                                  # prior std 0.4
A = torch.eye(2)
Y_OBS = torch.tensor([0.0, 0.0])                  # equidistant from both modes
SIGMA_NOISE = 0.9                                 # broad enough to keep both lobes

N = 24
SIGMA_MAX = 2.0
SIGMA_MIN = 0.02
SEED = 6
NB = 500
N_CHAINS = 1
LIM = 3.0

CMAP = LinearSegmentedColormap.from_list(
    "cooling", ["#dfe7f0", "#a8bcd4", "#5a7aa3", L.SAMPLE])


def gmm_log_prob(x, sigma):
    var = PRIOR_VAR + sigma ** 2
    diff = x.unsqueeze(1) - MODES.unsqueeze(0)
    expo = -0.5 * (diff ** 2).sum(-1) / var
    return torch.logsumexp(LOG_W + (-np.log(2 * np.pi * var)) + expo, dim=1)


def prior_score(x, sigma):
    x_ = x.detach().requires_grad_(True)
    (s,) = torch.autograd.grad(gmm_log_prob(x_, sigma).sum(), x_)
    return s.detach()


def prior_score_graph(x, sigma):
    (s,) = torch.autograd.grad(gmm_log_prob(x, sigma).sum(), x, create_graph=True)
    return s


def tweedie(x, sigma):
    return x + sigma ** 2 * prior_score(x, sigma)


def posterior_grid(n=280):
    g = np.linspace(-LIM, LIM, n)
    xx, yy = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([xx, yy], -1).reshape(-1, 2))
    prior_lp = gmm_log_prob(pts, torch.tensor(0.0))
    res = Y_OBS.unsqueeze(0) - pts @ A.T
    lp = prior_lp - 0.5 * (res ** 2).sum(-1) / SIGMA_NOISE ** 2
    lp = lp.numpy().reshape(xx.shape)
    return xx, yy, np.exp(lp - lp.max())


def _sigmas():
    return torch.tensor(np.geomspace(SIGMA_MAX, SIGMA_MIN, N + 1))


def _fac(s):
    s = s.numpy()
    return 2.0 * s[:-1] * (s[:-1] - s[1:])
def dps_chain(seed, guidance_scale=1.0):
    torch.manual_seed(seed)
    sigmas = _sigmas(); fac = _fac(sigmas)
    x = torch.randn(NB, 2) * SIGMA_MAX
    path = [x.clone()]
    for i in range(N):
        sig = sigmas[i].item()
        x_cur = x.detach().requires_grad_(True)
        x0 = x_cur + sig ** 2 * prior_score_graph(x_cur, sig)
        residual = x0 @ A.T - Y_OBS
        loss = (residual ** 2).sum()
        (ll_grad,) = torch.autograd.grad(loss, x_cur)
        ll_grad = ll_grad * 0.5 / (loss.detach().sqrt() + 1e-12)
        sc = (x0.detach() - x_cur.detach()) / sig ** 2
        x = (x_cur.detach() + fac[i] * sc + np.sqrt(fac[i]) * torch.randn_like(x)
             - guidance_scale * ll_grad.detach())
        path.append(x.clone())
    return torch.stack(path).numpy()


def _reverse_ode(x, sigma_start, steps=8):
    sigmas = torch.tensor(np.geomspace(sigma_start, 1e-3, steps + 1))
    fac = _fac(sigmas)
    for k in range(steps):
        x0 = tweedie(x, sigmas[k].item())
        x = x + 0.5 * fac[k] * (x0 - x) / sigmas[k].item() ** 2
    return x.detach()


def _langevin_inner(x0_hat, sigma, steps, lr_frac=0.2):
    eta = lr_frac * min(sigma ** 2, SIGMA_NOISE ** 2)
    x = x0_hat.clone()
    for _ in range(steps):
        data_grad = (Y_OBS.unsqueeze(0) - x @ A.T) @ A / SIGMA_NOISE ** 2
        prior_grad = (x0_hat - x) / sigma ** 2
        x = x + eta * (data_grad + prior_grad) + np.sqrt(2 * eta) * torch.randn_like(x)
    return x.detach()


def daps_chain(seed, ode_steps=8, lgvd_steps=40):
    torch.manual_seed(seed)
    sigmas = _sigmas()
    x = torch.randn(NB, 2) * SIGMA_MAX
    path = [x.clone()]
    for i in range(N):
        sigma = sigmas[i].item()
        sigma_next = sigmas[i + 1].item()
        x0_hat = _reverse_ode(x, sigma, ode_steps)
        x0y = _langevin_inner(x0_hat, sigma, lgvd_steps)
        x = x0y + sigma_next * torch.randn_like(x0y)
        path.append(x.clone())
    return torch.stack(path).numpy()
def posterior_lines(ax):
    xx, yy, dens = posterior_grid()
    lv = np.quantile(dens[dens > dens.max() * 1e-4], [0.5, 0.78, 0.94])
    ax.contour(xx, yy, dens, levels=np.r_[lv, dens.max()],
               colors=L.MODE_EDGE, linewidths=0.4, alpha=0.6)


def graded_chain(ax, path, k):
    p = path[:, k, :]
    T = len(p)
    seg = np.stack([p[:-1], p[1:]], axis=1)
    lc = LineCollection(seg, cmap=CMAP, array=np.linspace(0, 1, T - 1),
                        linewidths=1.0, alpha=0.85, zorder=5)
    ax.add_collection(lc)
    ax.scatter(p[:, 0], p[:, 1], c=np.linspace(0, 1, T), cmap=CMAP, s=15,
               zorder=6, linewidths=0)
    ax.plot([p[0, 0]], [p[0, 1]], "o", mfc="white", mec=L.SUBTLE, ms=7, mew=1.0,
            zorder=7)
    if k == 0:
        ax.annotate("start", xy=(p[0, 0], p[0, 1]), xytext=(8, -10),
                    textcoords="offset points", fontsize=13, color=L.INK)


def panel(ax, path, title, note):
    posterior_lines(ax)
    final = path[-1]
    ax.scatter(final[:, 0], final[:, 1], s=3, c=L.SAMPLE, alpha=0.12,
               linewidths=0, rasterized=True, zorder=3)
    for k in range(N_CHAINS):
        graded_chain(ax, path, k)
    ax.set_title(title, fontsize=19, loc="left")
    ax.text(0.04, 0.05, note, transform=ax.transAxes, fontsize=15, color=L.INK,
            va="bottom")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=13, length=3)
    ax.set_xlabel("$x_1$  (modes split along here)", fontsize=16)


def main():
    import argparse
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()
    L.apply_style()

    dps = dps_chain(SEED)
    daps = daps_chain(SEED)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 5.2), sharey=True,
                                 constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))
    panel(a1, dps, "coupled  (DPS)",
          "consecutive steps close:\nsmooth, local path")
    panel(a2, daps, "decoupled  (DAPS)",
          "consecutive steps far:\nnon-local jumps")
    a1.set_ylabel("$x_2$  (no mode split)", fontsize=16)

    sm = plt.cm.ScalarMappable(cmap=CMAP)
    cb = fig.colorbar(sm, ax=(a1, a2), fraction=0.035, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels([r"$\sigma_{\max}$", r"$\sigma_{\min}$"])
    cb.ax.tick_params(labelsize=13, length=0)
    cb.set_label("noise scale (cooling)", fontsize=13.5)
    cb.outline.set_linewidth(0.4)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", mfc="white", mec=L.SUBTLE, ms=8,
               label="broad-noise start"),
        Line2D([], [], color=L.MODE_EDGE, lw=1.0,
               label=r"posterior $p(x\mid y)$ (2 lobes)"),
        Line2D([], [], color=L.SAMPLE, lw=1.6, label="one example chain (cooling)"),
        Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=5, alpha=0.3,
               label="final samples (500 chains)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.010, 0.965),
               ncol=4, frameon=False, fontsize=13, handletextpad=0.5,
               columnspacing=1.5, labelcolor=L.INK, borderaxespad=0.0)
    out = f"{args.out_dir}/fig_decoupling"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
