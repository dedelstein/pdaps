"""Empirical annealing trajectories of the four samplers (Sec. 4.3-4.6).

Companion to fig_sampler_structure (a schematic of ONE step): each sampler is run
for real and a few walkers are drawn funnelling from broad noise to a SINGLE
target -- mirroring fig_gd_conditioning (one minimiser, one orange +). The toy is
deliberately unimodal so there is one unambiguous target; the bimodal-ambiguity
story lives in the density-grid figure, not here.

The target is anisotropic on purpose: A measures x1 (stiff, tight posterior) and
leaves x2 in null(A) (flat, loose posterior), so the posterior is an ellipse
elongated along x2. This is the sampler analogue of fig_gd_conditioning's
ill-conditioned valley: ISOTROPIC inner steps (DPS, DAPS) must creep along the
stiff axis, while PRECONDITIONED steps (pULA, pDAPS) align to the curvature and
travel straight. That preconditioning contrast -- not decoupling -- is what a 2-D
problem can honestly show.

Legibility moves over the DAPS paper's tangle: the CLEAN-state trajectory (not the
noisy iterate); a few walkers from a SHARED start; cooling encoded as a single-hue
light->dark-blue gradient along each path.

Self-contained: analytic unimodal-Gaussian prior + numpy ports of the four
samplers. Imports landscape only for the shared visual language.

Run:
  ./.venv/bin/python3 thesis_figures/fig_sampler_trajectories.py             # quad
  ./.venv/bin/python3 thesis_figures/fig_sampler_trajectories.py --mode singles
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import landscape as L
MODE = np.array([0.6, 0.6])        # the one prior mode == the target (orange +)
PRIOR_VAR = 0.25                   # prior std 0.5
S_MEAS = 1.6                       # singular value on x1 (measured, stiff)
A = np.diag([S_MEAS, 0.0])         # x2 is unmeasured -> null(A)
SIGMA_NOISE = 0.16                 # softens the ellipse so it reads as an ellipse
Y_OBS = A @ MODE                   # noiseless observation

N = 18
SIGMA_MAX = 1.8
SIGMA_MIN = 0.02
SEED = 0
N_WALK = 1
NB = 200
LIMX, LIMY = 2.6, 3.4

CMAP = LinearSegmentedColormap.from_list(
    "cooling", ["#dfe7f0", "#a8bcd4", "#5a7aa3", L.SAMPLE])   # light/hot -> dark/cold

METHODS = {
    "dps":   ("DPS",   r"coupled $\cdot$ isotropic"),
    "daps":  ("DAPS",  r"decoupled $\cdot$ isotropic"),
    "pula":  ("pULA",  r"coupled $\cdot$ preconditioned"),
    "pdaps": ("pDAPS", r"decoupled $\cdot$ preconditioned"),
}
def score(x, sigma):
    return (MODE - x) / (PRIOR_VAR + sigma ** 2)


def tweedie(x, sigma):
    return x + sigma ** 2 * score(x, sigma)


def sigmas_of(n, smax=SIGMA_MAX, smin=SIGMA_MIN):
    return np.geomspace(smax, smin, n + 1)


def factor_steps(s):
    return 2.0 * s[:-1] * (s[:-1] - s[1:])


class Rec:
    def __init__(self):
        self.paths = []

    def record(self, state):
        self.paths.append(state[:N_WALK].copy())

    def finish(self):
        return np.stack(self.paths, 0)            # (T, N_WALK, 2)


def _reverse_ode(x, sigma_start, steps=8):
    s = sigmas_of(steps, sigma_start, 1e-3)
    fac = factor_steps(s)
    for k in range(steps):
        x0 = tweedie(x, s[k])
        x = x + 0.5 * fac[k] * (x0 - x) / s[k] ** 2
    return x
def run_dps(rng, guidance=0.6):
    s = sigmas_of(N); fac = factor_steps(s)
    x = rng.standard_normal((NB, 2)) * SIGMA_MAX
    rec = Rec(); rec.record(x)                        # broad start
    for i in range(N):
        sig = s[i]
        x0 = tweedie(x, sig)
        residual = x0 @ A.T - Y_OBS
        loss = (residual ** 2).sum()
        c = PRIOR_VAR / (PRIOR_VAR + sig ** 2)        # d x0 / d x (scalar)
        ll_grad = c * (residual @ A) / (np.sqrt(loss) + 1e-12)
        x = (x + fac[i] * score(x, sig)
             + np.sqrt(fac[i]) * rng.standard_normal((NB, 2))
             - guidance * ll_grad)
        rec.record(x)                                 # noisy annealing iterate
    return rec.finish()


def _langevin_iso(x0_hat, sigma, steps, lr, rng):
    x = x0_hat.copy()
    for _ in range(steps):
        data_grad = (Y_OBS - x @ A.T) @ A / SIGMA_NOISE ** 2
        prior_grad = (x0_hat - x) / sigma ** 2
        x = x + lr * (data_grad + prior_grad) + np.sqrt(2 * lr) * rng.standard_normal((NB, 2))
    return x


def run_daps(rng, lgvd_steps=30, lr=2e-3):
    s = sigmas_of(N)
    x = rng.standard_normal((NB, 2)) * SIGMA_MAX
    rec = Rec(); rec.record(x)
    for i in range(N):
        x0_hat = _reverse_ode(x, s[i])
        x0y = _langevin_iso(x0_hat, s[i], lgvd_steps, lr, rng)
        x = x0y + s[i + 1] * rng.standard_normal((NB, 2))
        rec.record(x)
    return rec.finish()


def run_pula(rng, step_size=0.4, nb_lgv=6):
    A_s = A / SIGMA_NOISE
    y_s = Y_OBS / SIGMA_NOISE
    AtA = A_s.T @ A_s
    var = sigmas_of(N - 1) ** 2
    eye = np.eye(2)
    x = rng.standard_normal((NB, 2)) * SIGMA_MAX
    rec = Rec(); rec.record(x)
    for i in range(N):
        sigma = np.sqrt(var[i])
        M = np.linalg.inv(AtA + eye / var[i])
        for _ in range(nb_lgv):
            sc = score(x, sigma)
            residual = y_s - x @ A_s.T
            conv = (M @ (sc + residual @ A_s).T).T
            n1 = rng.standard_normal((A_s.shape[0], NB))
            n2 = rng.standard_normal((2, NB))
            z = (M @ (A_s.T @ n1 + n2 / np.sqrt(var[i]))).T
            x = x + 0.5 * step_size * conv + np.sqrt(step_size) * z
        rec.record(x)
    return rec.finish()


def _pdaps_inner(x0_hat, sigma, steps, gamma, tau, rng):
    lam_raw = 1.0 / sigma ** 2
    lam_target = lam_raw * tau ** 2
    A_s = A / SIGMA_NOISE
    y_s = Y_OBS / SIGMA_NOISE
    AtA = A_s.T @ A_s
    M = np.linalg.inv(AtA + lam_raw * np.eye(2))
    x = x0_hat.copy()
    for _ in range(steps):
        residual = y_s - x @ A_s.T
        grad = residual @ A_s - (x - x0_hat) * lam_target
        x = x + 0.5 * gamma * (M @ grad.T).T
        n1 = rng.standard_normal((A_s.shape[0], NB))
        n2 = rng.standard_normal((2, NB))
        z = (M @ (A_s.T @ n1 + np.sqrt(lam_raw) * n2)).T
        x = x + np.sqrt(gamma) * z
    return x


def run_pdaps(rng, lgvd_steps=12, step_size=0.4, tau=0.15):
    s = sigmas_of(N)
    x = rng.standard_normal((NB, 2)) * SIGMA_MAX
    rec = Rec(); rec.record(x)
    for i in range(N):
        x0_hat = _reverse_ode(x, s[i])
        x0y = _pdaps_inner(x0_hat, s[i], lgvd_steps, step_size, tau, rng)
        x = x0y + s[i + 1] * rng.standard_normal((NB, 2))
        rec.record(x)
    return rec.finish()


RUNNERS = {"dps": run_dps, "daps": run_daps, "pula": run_pula, "pdaps": run_pdaps}


def run_method(method):
    return RUNNERS[method](np.random.default_rng(SEED))
def posterior_contours(ax):
    """Thin contour LINES of the (Gaussian) posterior -- an ellipse tight on x1
    (measured) and loose on x2 (null), like fig_gd_conditioning's valley."""
    g = 240
    xs = np.linspace(-LIMX, LIMX, g)
    ys = np.linspace(-LIMY, LIMY, g)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx, yy], -1)
    quad_prior = ((pts - MODE) ** 2).sum(-1) / PRIOR_VAR
    res = pts @ A.T - Y_OBS
    quad_lik = (res ** 2).sum(-1) / SIGMA_NOISE ** 2
    nll = 0.5 * (quad_prior + quad_lik)
    nll -= nll.min()
    lv = np.geomspace(0.3, nll.max() * 0.9, 7)
    ax.contour(xx, yy, nll, levels=lv, colors=L.MODE_EDGE, linewidths=0.4, alpha=0.6)


def draw_graded_walkers(ax, paths):
    T = paths.shape[0]
    K = min(N_WALK, paths.shape[1])
    for k in range(K):
        p = paths[:, k, :]
        seg = np.stack([p[:-1], p[1:]], axis=1)
        lc = LineCollection(seg, cmap=CMAP, array=np.linspace(0, 1, T - 1),
                            linewidths=1.0, alpha=0.7, zorder=4)
        ax.add_collection(lc)
        ax.scatter(p[:, 0], p[:, 1], c=np.linspace(0, 1, T), cmap=CMAP,
                   s=14, zorder=5, linewidths=0)
        ax.plot([p[0, 0]], [p[0, 1]], "o", mfc="white", mec=L.SUBTLE, ms=6.5,
                mew=1.0, zorder=6)                        # start (hot)
        ax.plot([p[-1, 0]], [p[-1, 1]], "o", mfc=L.SAMPLE, mec="white",
                ms=6, mew=0.7, zorder=7)                  # end (cold)


def panel(ax, paths, title, subtitle):
    posterior_contours(ax)
    draw_graded_walkers(ax, paths)
    ax.scatter([MODE[0]], [MODE[1]], marker="x", s=110, c=L.TRUTH,
               linewidths=1.8, zorder=8)
    ax.set_xlim(-LIMX, LIMX)
    ax.set_ylim(-LIMY, LIMY)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=13, length=3)
    if title:
        ax.set_title(title, fontsize=19, loc="left")
    if subtitle:
        ax.text(0.03, 0.04, subtitle, transform=ax.transAxes, fontsize=13.5,
                color=L.INK, va="bottom", style="italic")


CAPTION = (r"gray $= p(x\mid y)$ (darker = larger) $\quad$ "
           r"$\times$ = true signal $x^\star$ $\quad$ $\circ$ = broad-noise start $\quad$ "
           r"path = walker's clean-state estimate, light $\rightarrow$ dark blue (cooling)")


def _save(fig, out):
    import matplotlib.pyplot as plt
    fig.savefig(f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")


def fig_quad(out):
    import matplotlib.pyplot as plt
    order = ["dps", "pula", "daps", "pdaps"]   # coupled row / decoupled row
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 8.6), sharex=True, sharey=True)
    for a, m in zip(ax.flat, order):
        title, sub = METHODS[m]
        panel(a, run_method(m), title, sub)
    for a in ax[:, 0]:
        a.set_ylabel("$x_2$  (unmeasured / null)", fontsize=16)
    for a in ax[1, :]:
        a.set_xlabel("$x_1$  (measured)", fontsize=16)
    fig.text(0.012, 0.93, CAPTION, fontsize=13.5, color=L.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, out)


def fig_singles(out_dir):
    import matplotlib.pyplot as plt
    for m in METHODS:
        title, sub = METHODS[m]
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        panel(ax, run_method(m), title, sub)
        ax.set_xlabel("$x_1$  (measured)", fontsize=16)
        ax.set_ylabel("$x_2$  (unmeasured / null)", fontsize=16)
        fig.tight_layout()
        _save(fig, f"{out_dir}/fig_traj_{m}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    ap.add_argument("--mode", choices=["quad", "singles"], default="quad")
    args = ap.parse_args()
    L.apply_style()
    if args.mode == "quad":
        fig_quad(f"{args.out_dir}/fig_sampler_trajectories")
    else:
        fig_singles(args.out_dir)


if __name__ == "__main__":
    main()
