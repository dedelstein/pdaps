"""Shared 2-D landscape, samplers, and visual language for the thesis figures.

This module is the single source of the *progressive visual identity* used by
every pedagogical figure in the methods narrative (Ch. 2-4):

  * one palette + glyph vocabulary (true signal x* = orange x, samples = blue,
    density = gray tints, modes = open circles);
  * one framing convention (range-frame, in-panel direct labels, serif, ~1.5:1);
  * one "landscape" motif.

The sampler problem is a deliberately legible 2-D inverse problem:

  * a 2-mode Gaussian prior whose modes are split along the NULL axis of A;
  * a rank-deficient, stiff forward operator A = diag([s_meas, 0]) so the x-axis
    is well measured (high curvature) and the y-axis is unmeasured (flat);
  * a small measurement noise so the posterior is finite and bimodal in y.

The six samplers are faithful 2-D ports of toy_2d.py (run_dps/run_daps/run_pula/
run_pdaps). Each records, per outer level: a clean-state estimate (for walker
trajectories) and the mean range/null component norms (for the freeze plots).

Plot-only. Does NOT import the 64-D research harness.
"""

from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
TRUTH = "#d1701a"      # orange  -- the true signal x* (x marker)
SAMPLE = "#1f4f9e"     # deep but clearly-blue -- sampler output / walkers
MODE_EDGE = "#555555"  # open-circle posterior modes
INK = "#000000"        # black -- ALL text (titles->captions aside) and structure
SUBTLE = "#777777"     # gray -- faint reference LINES only (never text)
DENSITY = ["#f0f0f0", "#dcdcdc", "#c0c0c0"]   # gray tints, light -> dark


def apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{xcolor}"
            r"\definecolor{cblue}{HTML}{1F4F9E}\definecolor{corange}{HTML}{D1701A}"),
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "savefig.dpi": 300,
    })
MEAS_CENTER = 1.0
MODES = torch.tensor([[MEAS_CENTER, 2.0], [MEAS_CENTER, -2.0]])
LOG_W = torch.log(torch.full((MODES.shape[0],), 1.0 / MODES.shape[0]))
CLUSTER_STD = 0.5
PRIOR_VAR = CLUSTER_STD ** 2

S_MEAS = 2.0          # singular value on the measured (x) axis -> stiff
S_NULL = 0.0          # singular value on the null (y) axis     -> unmeasured
A = torch.diag(torch.tensor([S_MEAS, S_NULL]))
SIGMA_NOISE = 0.1

X_TRUE = torch.tensor([MEAS_CENTER, 2.0])          # lives in the +y mode
Y_OBS = A @ X_TRUE                                  # noiseless observation -> (0, 0)

LIM_X, LIM_Y = 2.6, 3.4
def gmm_log_prob(x, sigma, modes=MODES, log_w=LOG_W, prior_var=PRIOR_VAR):
    var = prior_var + sigma ** 2
    diff = x.unsqueeze(1) - modes.unsqueeze(0)            # (B, K, 2)
    expo = -0.5 * (diff ** 2).sum(-1) / var
    log_norm = -np.log(2 * np.pi * var)
    return torch.logsumexp(log_w + log_norm + expo, dim=1)


def prior_score(x, sigma):
    x_ = x.detach().requires_grad_(True)
    lp = gmm_log_prob(x_, sigma).sum()
    (s,) = torch.autograd.grad(lp, x_)
    return s.detach()


def prior_score_graph(x, sigma):
    lp = gmm_log_prob(x, sigma).sum()
    (s,) = torch.autograd.grad(lp, x, create_graph=True)
    return s


def tweedie(x, sigma):
    """VE posterior-mean estimate E[x0 | x] = x + sigma^2 * score."""
    return x + sigma ** 2 * prior_score(x, sigma)
def posterior_log_prob(x):
    prior_lp = gmm_log_prob(x, sigma=0.0)
    residual = Y_OBS.unsqueeze(0) - x @ A.T
    log_lik = -0.5 * (residual ** 2).sum(-1) / SIGMA_NOISE ** 2
    return prior_lp + log_lik


def posterior_density_grid(n=300):
    xs = np.linspace(-LIM_X, LIM_X, n)
    ys = np.linspace(-LIM_Y, LIM_Y, n)
    xx, yy = np.meshgrid(xs, ys)
    pts = torch.tensor(np.stack([xx, yy], -1).reshape(-1, 2))
    logp = posterior_log_prob(pts).numpy().reshape(xx.shape)
    logp -= logp.max()
    return xx, yy, np.exp(logp)


def projectors():
    """Orthogonal projectors onto range(A^T) (measured) and null(A) (unmeasured)."""
    U, S, Vh = torch.linalg.svd(A)
    V = Vh.T
    tol = S.max() * 1e-8
    par_cols = V[:, S > tol]
    P_par = par_cols @ par_cols.T if par_cols.shape[1] else torch.zeros(2, 2)
    P_perp = torch.eye(2) - P_par
    return P_par, P_perp


P_PAR, P_PERP = projectors()


def components(X):
    """Mean ||range component|| and ||null component|| over a batch."""
    par = (X @ P_PAR.T)
    perp = (X @ P_PERP.T)
    return float(par.norm(dim=1).mean()), float(perp.norm(dim=1).mean())


def mode_mass(final):
    """Fraction of in-view samples in the +y mode vs the -y mode."""
    m = (np.abs(final[:, 0]) < LIM_X) & (np.abs(final[:, 1]) < LIM_Y)
    f = final[m]
    if len(f) == 0:
        return 0.0, 0.0
    top = float((f[:, 1] > 0).mean())
    return top, 1.0 - top
def make_sigmas(N, sigma_max, sigma_min):
    return torch.tensor(np.geomspace(sigma_max, sigma_min, N + 1))


def factor_steps(sigmas):
    s = sigmas.numpy()
    return 2.0 * s[:-1] * (s[:-1] - s[1:])


def init_broad(nb, sigma_max):
    return torch.randn(nb, 2) * sigma_max


def init_adversarial(nb, sigma_max):
    """All mass in the wrong mode (the one furthest from x_true)."""
    dists = torch.linalg.norm(MODES - X_TRUE, dim=1)
    wrong = MODES[int(torch.argmax(dists))]
    return wrong + CLUSTER_STD * torch.randn(nb, 2)


class _Recorder:
    """Collects clean-state estimates per outer level for paths + freeze traces."""

    def __init__(self, n_paths=6):
        self.n_paths = n_paths
        self.sigmas, self.par, self.perp, self.paths = [], [], [], []

    def record(self, sigma, x0, state=None):
        if state is None:
            state = x0
        p, q = components(x0)
        self.sigmas.append(sigma)
        self.par.append(p)
        self.perp.append(q)
        self.paths.append(state[:self.n_paths].detach().clone().numpy())

    def finish(self, final):
        return {
            "final": final.detach().numpy(),
            "sigmas": np.array(self.sigmas),
            "par": np.array(self.par),
            "perp": np.array(self.perp),
            "paths": np.stack(self.paths, 0) if self.paths else None,  # (T, K, 2)
        }
def run_gradient_descent(nb, lr=0.003, steps=700, n_paths=12):
    """MAP optimisation: ascend the posterior log-density. Point estimates."""
    x = init_broad(nb, LIM_X)
    rec = _Recorder(n_paths)
    for _ in range(steps):
        ps = prior_score(x, 1e-3)
        residual = Y_OBS.unsqueeze(0) - x @ A.T
        data_grad = (residual @ A) / SIGMA_NOISE ** 2
        x = x + lr * (ps + data_grad)
        rec.record(1e-3, x)
    return rec.finish(x)


def run_unconditional(nb, N, sigma_max, sigma_min):
    """Reverse-diffusion sampling of the prior alone (no data)."""
    sigmas = make_sigmas(N, sigma_max, sigma_min)
    fac = factor_steps(sigmas)
    x = init_broad(nb, sigma_max)
    rec = _Recorder()
    for i in range(N):
        sig = sigmas[i].item()
        x = x + fac[i] * prior_score(x, sig) + np.sqrt(fac[i]) * torch.randn_like(x)
        rec.record(sig, tweedie(x, max(sig, 1e-3)))
    return rec.finish(x)


def run_dps(nb, N, sigma_max, sigma_min, init, guidance_scale=1.0):
    sigmas = make_sigmas(N, sigma_max, sigma_min)
    fac = factor_steps(sigmas)
    x = init(nb, sigma_max)
    rec = _Recorder()
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
        rec.record(sig, x0.detach())
    return rec.finish(x)


def _reverse_ode(x, sigma_start, num_steps=8):
    sigmas = make_sigmas(num_steps, sigma_start, 1e-3)
    fac = factor_steps(sigmas)
    for k in range(num_steps):
        s = sigmas[k].item()
        x0 = tweedie(x, s)
        x = x + 0.5 * fac[k] * (x0 - x) / s ** 2
    return x.detach()


def _langevin_inner(x0_hat, sigma, steps, lr):
    x = x0_hat.clone()
    for _ in range(steps):
        residual = Y_OBS.unsqueeze(0) - x @ A.T
        data_grad = (residual @ A) / SIGMA_NOISE ** 2
        prior_grad = (x0_hat - x) / sigma ** 2
        x = x + lr * (data_grad + prior_grad) + np.sqrt(2 * lr) * torch.randn_like(x)
    return x.detach()


def run_daps(nb, N, sigma_max, sigma_min, init, ode_steps=8, lgvd_steps=30, lr=2e-3):
    sigmas = make_sigmas(N, sigma_max, sigma_min)
    x = init(nb, sigma_max)
    rec = _Recorder()
    for step in range(N):
        sigma = sigmas[step].item()
        sigma_next = sigmas[step + 1].item()
        x0_hat = _reverse_ode(x, sigma, ode_steps)
        x0y = _langevin_inner(x0_hat, sigma, lgvd_steps, lr)
        x = x0y + sigma_next * torch.randn_like(x0y)
        rec.record(sigma, x0y)
    return rec.finish(x)


def run_pula(nb, N, sigma_max, sigma_min, init, step_size=0.4, nb_langevin=6,
             warm_start=True):
    A_s = A / SIGMA_NOISE
    y_s = Y_OBS / SIGMA_NOISE
    AtA = A_s.T @ A_s
    variances = make_sigmas(N - 1, sigma_max, sigma_min) ** 2
    eye = torch.eye(2)
    M0 = torch.inverse(AtA + eye / variances[0])
    if init is init_adversarial or not warm_start:
        x = init(nb, sigma_max)
    else:
        mean = M0 @ (A_s.T @ y_s)
        x = mean + torch.randn(nb, 2) @ torch.linalg.cholesky(M0).T
    rec = _Recorder()
    for i in range(N):
        var = variances[i].item()
        sigma = np.sqrt(var)
        M = torch.inverse(AtA + eye / var)
        for _ in range(nb_langevin):
            sc = prior_score(x, sigma)
            residual = y_s - x @ A_s.T
            conv_grad = (M @ (sc + residual @ A_s).T).T
            n1 = torch.randn(A_s.shape[0], nb)
            n2 = torch.randn(2, nb)
            z = (M @ (A_s.T @ n1 + n2 / np.sqrt(var))).T
            x = x + 0.5 * step_size * conv_grad + np.sqrt(step_size) * z
        rec.record(sigma, x)
    return rec.finish(x)


def _pdaps_inner(x0_hat, sigma, steps, gamma, tau):
    lam_raw = 1.0 / sigma ** 2
    lam_target = lam_raw * tau ** 2
    A_s = A / SIGMA_NOISE
    y_s = Y_OBS / SIGMA_NOISE
    AtA = A_s.T @ A_s
    M = torch.inverse(AtA + lam_raw * torch.eye(2))
    x = x0_hat.clone()
    for _ in range(steps):
        residual = y_s - x @ A_s.T
        grad = residual @ A_s - (x - x0_hat) * lam_target
        x = x + 0.5 * gamma * (M @ grad.T).T
        n1 = torch.randn(A_s.shape[0], x.shape[0])
        n2 = torch.randn(2, x.shape[0])
        z = (M @ (A_s.T @ n1 + np.sqrt(lam_raw) * n2)).T
        x = x + np.sqrt(gamma) * z
    return x.detach()


def run_pdaps(nb, N, sigma_max, sigma_min, init, ode_steps=8, lgvd_steps=12,
              step_size=0.4, tau=0.15):
    sigmas = make_sigmas(N, sigma_max, sigma_min)
    x = init(nb, sigma_max)
    rec = _Recorder()
    for step in range(N):
        sigma = sigmas[step].item()
        sigma_next = sigmas[step + 1].item()
        x0_hat = _reverse_ode(x, sigma, ode_steps)
        x0y = _pdaps_inner(x0_hat, sigma, lgvd_steps, step_size, tau)
        x = x0y + sigma_next * torch.randn_like(x0y)
        rec.record(sigma, x0y)
    return rec.finish(x)
def frame_landscape(ax, xlim=None, ylim=None):
    ax.set_xlim(*(xlim or (-LIM_X, LIM_X)))
    ax.set_ylim(*(ylim or (-LIM_Y, LIM_Y)))
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=8, length=3)
    ax.set_aspect("equal")


def frame_to_data(ax, arrays, pad=0.10, qlo=0.5, qhi=99.5):
    """Range-frame: zoom the axes to the data extent (robust percentiles) so a
    narrow distribution fills the panel and the AXIS NUMBERS report its narrowness,
    rather than burying it in a thin slice of a wide equal-aspect frame. Aspect is
    deliberately NOT equalised -- the measured axis is genuinely much narrower than
    the unmeasured one, and that asymmetry is the point."""
    pts = np.concatenate([np.asarray(a).reshape(-1, 2)
                          for a in arrays if a is not None], axis=0)
    pts = pts[np.isfinite(pts).all(1)]
    xlo, xhi = np.percentile(pts[:, 0], [qlo, qhi])
    ylo, yhi = np.percentile(pts[:, 1], [qlo, qhi])
    dx, dy = (xhi - xlo) or 1.0, (yhi - ylo) or 1.0
    ax.set_xlim(xlo - pad * dx, xhi + pad * dx)
    ax.set_ylim(ylo - pad * dy, yhi + pad * dy)
    ax.tick_params(labelsize=8, length=3)


def draw_posterior_background(ax, grid=None):
    xx, yy, dens = grid if grid is not None else posterior_density_grid()
    levels = np.quantile(dens[dens > dens.max() * 1e-4], [0.5, 0.8, 0.95])
    ax.contourf(xx, yy, dens, levels=np.r_[levels, dens.max()],
                colors=DENSITY, antialiased=True)


def draw_modes(ax):
    ax.scatter(MODES[:, 0], MODES[:, 1], s=70, facecolors="none",
               edgecolors=MODE_EDGE, linewidths=0.8, zorder=5)


def draw_truth(ax):
    ax.scatter([X_TRUE[0]], [X_TRUE[1]], marker="x", s=90, c=TRUTH,
               linewidths=1.4, zorder=6)


def draw_samples(ax, final, n_show=700, rng=0, s=2.2, alpha=0.30):
    idx = np.random.default_rng(rng).permutation(len(final))[:n_show]
    ax.scatter(final[idx, 0], final[idx, 1], s=s, c=SAMPLE,
               alpha=alpha, linewidths=0, rasterized=True)


def draw_walkers(ax, paths, n=4):
    """A few annealing trajectories. Faint connecting lines carry the step order;
    the per-step dots (the actual iterate at each noise level) carry the eye as the
    walker funnels from broad noise into a mode. Coupled methods step smoothly;
    decoupled methods make non-local jumps -- both stay legible this way."""
    if paths is None:
        return
    K = min(n, paths.shape[1])
    for k in range(K):
        p = paths[:, k, :]
        ax.plot(p[:, 0], p[:, 1], "-", color=SAMPLE, lw=0.5, alpha=0.35, zorder=4)
        ax.plot(p[:, 0], p[:, 1], ".", color=SAMPLE, ms=3.2, alpha=0.85, zorder=4)
        ax.plot([p[0, 0]], [p[0, 1]], "o", mfc="white", mec=SAMPLE, ms=6,
                mew=1.0, zorder=5)
        ax.plot([p[-1, 0]], [p[-1, 1]], "o", mfc=SAMPLE, mec=SAMPLE, ms=4,
                zorder=5)


def annotate_mass(ax, final):
    top, bot = mode_mass(final)
    ax.text(0.03, 0.04, f"mode mass  {top:.2f} / {bot:.2f}",
            transform=ax.transAxes, fontsize=8.5, color=INK)


def range_null_panel(ax, res):
    """Plot ||range comp|| and ||null comp|| against noise scale sigma_t."""
    s = res["sigmas"]
    ax.plot(s, res["par"], color=SAMPLE, lw=1.4, label="range (measured)")
    ax.plot(s, res["perp"], color=TRUTH, lw=1.4, label="null (unmeasured)")
    ax.set_xscale("log")
    ax.invert_xaxis()           # schedule cools left -> right
    ax.set_xlabel(r"noise scale $\sigma_t$  (cooling $\rightarrow$)", fontsize=9)
    ax.tick_params(labelsize=8, length=3)
