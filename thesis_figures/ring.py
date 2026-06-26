"""Ring-manifold landscape for the annealing-bridge figure.

(Reconstructed helper module: fig_annealed_bridge.py imports it.)

A 2-D stand-in for the data manifold: the prior concentrates on a ring of
radius RADIUS (radial width WIDTH), and the measurement observes only the
x1 coordinate (the measured axis; x2 is the null axis). The truth sits on
the ring at (MEAS, +sqrt(RADIUS^2 - MEAS^2)) and the ring/band intersection
is symmetric in x2, so the posterior is bimodal along null(A) by geometry.

  * ring_log_prob(x, sigma)   : sigma-diffused ring prior (radial-Gaussian
                                approximation of the ring * Gaussian
                                convolution -- a blob at large sigma, a sharp
                                ring as sigma -> 0);
  * annealed_loglik(x, sigma) : measurement band x1 ~ MEAS whose variance
                                SIGMA_NOISE^2 + sigma^2 anneals with the
                                schedule;
  * density_grid / gray_density / draw_truth / draw_modes / frame : panel
                                helpers in the shared visual language.
"""

from __future__ import annotations

import numpy as np

import landscape as L

RADIUS = np.sqrt(5.0)        # ring through the landscape modes (1, +/-2)
WIDTH = 0.18                 # radial width of the ring prior
MEAS = 1.0                   # observed x1 coordinate
SIGMA_NOISE = 0.10           # measurement noise floor
LIM_X, LIM_Y = 2.6, 3.4
X_TRUE = np.array([MEAS, np.sqrt(RADIUS ** 2 - MEAS ** 2)])
MODES = np.array([X_TRUE, X_TRUE * np.array([1.0, -1.0])])


def ring_log_prob(x, sigma):
    r = np.linalg.norm(np.asarray(x), axis=-1)
    return -0.5 * (r - RADIUS) ** 2 / (WIDTH ** 2 + sigma ** 2)


def annealed_loglik(x, sigma):
    var = SIGMA_NOISE ** 2 + sigma ** 2
    return -0.5 * (np.asarray(x)[..., 0] - MEAS) ** 2 / var


def density_grid(log_fn, n=300):
    xs = np.linspace(-LIM_X, LIM_X, n)
    ys = np.linspace(-LIM_Y, LIM_Y, n)
    xx, yy = np.meshgrid(xs, ys)
    logp = log_fn(np.stack([xx, yy], -1))
    logp = logp - logp.max()
    return xx, yy, np.exp(logp)


def gray_density(ax, grid):
    xx, yy, dens = grid
    levels = np.quantile(dens[dens > dens.max() * 1e-4], [0.5, 0.8, 0.95])
    ax.contourf(xx, yy, dens, levels=np.r_[levels, dens.max()],
                colors=L.DENSITY, antialiased=True)


def draw_modes(ax):
    ax.scatter(MODES[:, 0], MODES[:, 1], s=70, facecolors="none",
               edgecolors=L.MODE_EDGE, linewidths=0.8, zorder=5)


def draw_truth(ax):
    ax.scatter([X_TRUE[0]], [X_TRUE[1]], marker="x", s=90, c=L.TRUTH,
               linewidths=1.4, zorder=6)


def frame(ax):
    ax.set_xlim(-LIM_X, LIM_X)
    ax.set_ylim(-LIM_Y, LIM_Y)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=8, length=3)
    ax.set_aspect("equal")
