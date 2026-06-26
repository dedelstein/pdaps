"""Real-data toy figures: DPS / DAPS / pULA / pDAPS on the actual toy_2d problem.

Unlike fig_decoupling / fig_preconditioning / fig_sampler_structure (which use
bespoke reimplementations on simplified landscapes), every panel here is driven
by the *research* samplers in toy_2d.py -- run_dps / run_daps / run_pula /
run_pdaps -- on the real scenarios (toy_a_mode_recovery, toy_b_stiffness): a
4-mode GMM prior with a near-singular forward operator A = diag(a_diag) whose
x2 axis is barely observed (the stiff / unmeasured direction).

Three figures, each for both scenarios (stacked rows):

  clouds       : posterior contours + real final sample cloud + one cooling-
                 coloured example chain + x* marker, per method. The honest
                 "characterise the mechanism" view (cf. FIGURE_STYLE).
  precond      : coverage of the unmeasured x2 axis -- the x2 marginal of each
                 method vs the ground-truth posterior marginal, grouped
                 isotropic (DPS/DAPS) vs preconditioned (pULA/pDAPS). The
                 dimension-independent preconditioning axis, which DOES show in 2D.
  convergence  : per-step metrics from the progress snapshots (data misfit and
                 x2 coverage) against the cooling schedule sigma_t.

Run:  ./.venv/bin/python3 thesis_figures/fig_toy_samplers.py
      ./.venv/bin/python3 thesis_figures/fig_toy_samplers.py --only clouds
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import toy_2d as T          # noqa: E402
import landscape as L       # noqa: E402

torch.set_default_dtype(torch.float64)

SCENARIOS = ["toy_a_mode_recovery", "toy_b_stiffness"]
SCEN_LABEL = {"toy_a_mode_recovery": "toy A  (mode recovery)",
              "toy_b_stiffness": "toy B  (stiffness)"}
METHODS = ["DPS", "DAPS", "pULA", "P-DAPS"]
ISO = ["DPS", "DAPS"]            # isotropic inner step
PRE = ["pULA", "P-DAPS"]        # preconditioned inner step

NB_CLOUD = 2000
SEED = 0
METRIC_SEEDS = [0, 1, 2, 3, 4]

CMAP = LinearSegmentedColormap.from_list(
    "cooling", ["#dfe7f0", "#a8bcd4", "#5a7aa3", L.SAMPLE])
GRAY = LinearSegmentedColormap.from_list(
    "dens", ["#ffffff", "#e0e0e0", "#bdbdbd", "#969696", "#636363"])
def run_all(scenario, nb=NB_CLOUD, seed=SEED):
    T.configure_scenario(scenario, quiet=True)
    rp = T.SCENARIOS[scenario]["run_params"]
    N = rp["N"]
    common = dict(nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
                  budget_steps=list(range(1, N + 1)))
    out = {}
    torch.manual_seed(seed)
    out["DPS"] = T.run_dps(guidance_scale=rp["dps_guidance_scale"], **common)
    torch.manual_seed(seed)
    out["DAPS"] = T.run_daps(ode_steps=rp["daps_ode_steps"],
                             lgvd_steps=rp["daps_langevin_steps"],
                             lgvd_lr=rp["daps_langevin_lr"], **common)
    torch.manual_seed(seed)
    out["pULA"] = T.run_pula(step_size=rp["pula_step_size"],
                             nb_langevin=rp["pula_nb_langevin"], **common)
    torch.manual_seed(seed)
    out["P-DAPS"] = T.run_pdaps_v3(ode_steps=rp["pdaps_ode_steps"], **common)
    geom = dict(lim=float(T.LIM), modes=T.MU.cpu().numpy(),
                x_true=T.X_TRUE.cpu().numpy(),
                A=T.A.cpu().numpy(), y=T.Y_OBS.cpu().numpy(),
                sigma_noise=float(T.SIGMA_NOISE),
                grid=_posterior_grid(), gt=T.sample_ground_truth(120_000))
    return out, geom, N


def _posterior_grid(n=260):
    xx, yy, pts = T.make_grid(n)
    logp = T.posterior_log_prob(pts).detach().cpu().numpy().reshape(xx.shape)
    logp -= logp.max()
    return xx, yy, np.exp(logp)


def _progress_paths(res, k):
    """Stack the first k walkers across all recorded steps -> (T, k, 2)."""
    steps = sorted(res["progress"])
    if not steps:
        return None
    return np.stack([res["progress"][s]["samples"][:k] for s in steps], 0)


def _progress_clouds(res):
    steps = sorted(res["progress"])
    return np.array([np.sqrt(s) for s in steps]), steps  # placeholder, see metrics
def draw_gt(ax, gt, n_show=3500, rng=1):
    """Faint gray cloud of ground-truth posterior samples. The posterior here is
    near-degenerate (tight modes, small noise), so a sample cloud reads where
    filled contours collapse to specks."""
    idx = np.random.default_rng(rng).permutation(len(gt))[:n_show]
    ax.scatter(gt[idx, 0], gt[idx, 1], s=4, c=L.DENSITY[2], alpha=0.5,
               linewidths=0, rasterized=True, zorder=1)


def draw_modes(ax, modes):
    ax.scatter(modes[:, 0], modes[:, 1], s=55, facecolors="none",
               edgecolors=L.MODE_EDGE, linewidths=0.7, zorder=5)


def draw_truth(ax, x_true):
    ax.scatter([x_true[0]], [x_true[1]], marker="x", s=80, c=L.TRUTH,
               linewidths=1.4, zorder=7)


def draw_cloud(ax, final, n_show=700, rng=0):
    idx = np.random.default_rng(rng).permutation(len(final))[:n_show]
    ax.scatter(final[idx, 0], final[idx, 1], s=2.2, c=L.SAMPLE, alpha=0.28,
               linewidths=0, rasterized=True, zorder=3)


def draw_cooling_chain(ax, path):
    """One example chain, segments coloured by noise scale (light=hot, dark=cold)."""
    if path is None:
        return
    p = path[:, 0, :]
    Tn = len(p)
    seg = np.stack([p[:-1], p[1:]], axis=1)
    lc = LineCollection(seg, cmap=CMAP, array=np.linspace(0, 1, Tn - 1),
                        linewidths=1.0, alpha=0.9, zorder=6)
    ax.add_collection(lc)
    ax.scatter(p[:, 0], p[:, 1], c=np.linspace(0, 1, Tn), cmap=CMAP, s=10,
               zorder=6, linewidths=0)
    ax.plot([p[0, 0]], [p[0, 1]], "o", mfc="white", mec=L.SUBTLE, ms=6, mew=1.0,
            zorder=7)


def frame(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=11, length=3)
def fig_clouds(data, out_dir):
    import matplotlib.pyplot as plt
    nrow = len(SCENARIOS)
    fig, axes = plt.subplots(nrow, 4, figsize=(11.0, 2.9 * nrow + 0.4),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, scen in enumerate(SCENARIOS):
        out, geom, _ = data[scen]
        for c, m in enumerate(METHODS):
            ax = axes[r, c]
            draw_gt(ax, geom["gt"])
            draw_modes(ax, geom["modes"])
            draw_cloud(ax, out[m]["final"])
            draw_cooling_chain(ax, _progress_paths(out[m], 1))
            draw_truth(ax, geom["x_true"])
            frame(ax, geom["lim"])
            if r == 0:
                ax.set_title(m, fontsize=17, loc="center")
            if c == 0:
                ax.set_ylabel(SCEN_LABEL[scen] + "\n$x_2$", fontsize=13)
            if r == nrow - 1:
                ax.set_xlabel("$x_1$", fontsize=13)
    sm = plt.cm.ScalarMappable(cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels([r"$\sigma_{\max}$", r"$\sigma_{\min}$"])
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_label("example chain: noise scale (cooling)", fontsize=12)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", mfc=L.DENSITY[2], mec="none", ms=7,
               label="posterior (ground truth)"),
        Line2D([], [], marker="o", ls="none", mfc=L.SAMPLE, mec="none", ms=6,
               alpha=0.6, label="final samples"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=L.MODE_EDGE, ms=8,
               label="prior modes"),
        Line2D([], [], color=L.SAMPLE, lw=1.6, label="example chain (cooling)"),
        Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=8, mew=1.6,
               label=r"truth $x^\star$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, frameon=False, fontsize=11, handletextpad=0.5,
               columnspacing=1.5)
    _save(fig, out_dir, "fig_toy_clouds")
def _density(log_prob_fn, n=260):
    xx, yy, d = T.density_on_grid(log_prob_fn, n)
    return xx, yy, d / d.max()


def draw_field(ax, xx, yy, d):
    dd = d ** 0.42          # gamma-compress so tight modes / tails stay visible
    ax.contourf(xx, yy, dd, levels=np.linspace(1e-3, 1.0, 16), cmap=GRAY,
                antialiased=True, zorder=0)


def fig_problem(out_dir, scenarios=SCENARIOS):
    import matplotlib.pyplot as plt
    nrow = len(scenarios)
    fig, axes = plt.subplots(nrow, 3, figsize=(9.5, 3.1 * nrow + 0.3),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    titles = [r"prior  $p(x)$",
              r"likelihood  $p(y\mid x)$" + "\n" + r"{\small (flat in $x_2$)}",
              r"posterior  $p(x\mid y)$"]
    for r, scen in enumerate(scenarios):
        T.configure_scenario(scen, quiet=True)
        lim = float(T.LIM)
        modes = T.MU.cpu().numpy()
        x_true = T.X_TRUE.cpu().numpy()
        A, y, sn = T.A, T.Y_OBS, T.SIGMA_NOISE
        fields = [
            _density(lambda x: T._gmm_log_prob(x, sigma=0.0)),
            _density(lambda x: -0.5 * ((y.unsqueeze(0) - (x @ A.T)) ** 2).sum(-1) / sn ** 2),
            _density(T.posterior_log_prob),
        ]
        for c, (xx, yy, d) in enumerate(fields):
            ax = axes[r, c]
            draw_field(ax, xx, yy, d)
            if c != 1:                              # modes on prior + posterior
                ax.scatter(modes[:, 0], modes[:, 1], s=55, facecolors="none",
                           edgecolors=L.MODE_EDGE, linewidths=0.7, zorder=4)
            if c == 2:                              # truth on the posterior
                ax.scatter([x_true[0]], [x_true[1]], marker="x", s=80, c=L.TRUTH,
                           linewidths=1.4, zorder=6)
            frame(ax, lim)
            if r == 0:
                ax.set_title(titles[c], fontsize=15)
            if r == nrow - 1:
                ax.set_xlabel("$x_1$  (measured)", fontsize=12)
            if c == 0:
                ax.set_ylabel(SCEN_LABEL[scen] + "\n$x_2$ (unmeasured)", fontsize=12)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=L.DENSITY[2], edgecolor="none",
              label="density (darker $=$ larger)"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=L.MODE_EDGE, ms=8,
               label="prior modes"),
        Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=8, mew=1.6,
               label=r"truth $x^\star$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=11, handletextpad=0.5,
               columnspacing=1.8)
    _save(fig, out_dir, "fig_toy_problem")
def _w1(a, b):
    """1-D Wasserstein-1 distance between two samples (quantile integral)."""
    qs = np.linspace(0, 1, 400)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _upper_mass(x2):
    return float((x2 > 0).mean())


def _finals_for_seed(scenario, seed, nb=NB_CLOUD):
    """Run the four samplers once at `seed` (no per-step snapshots) and return each
    method's x2 finals plus the ground-truth x2 marginal. Mirrors run_all's sampler
    settings; P-DAPS is the exact-split sampler at its default operating point."""
    T.configure_scenario(scenario, quiet=True)
    rp = T.SCENARIOS[scenario]["run_params"]
    common = dict(nb=nb, N=rp["N"], sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"])
    out = {}
    torch.manual_seed(seed)
    out["DPS"] = T.run_dps(guidance_scale=rp["dps_guidance_scale"], **common)
    torch.manual_seed(seed)
    out["DAPS"] = T.run_daps(ode_steps=rp["daps_ode_steps"], lgvd_steps=rp["daps_langevin_steps"],
                             lgvd_lr=rp["daps_langevin_lr"], **common)
    torch.manual_seed(seed)
    out["pULA"] = T.run_pula(step_size=rp["pula_step_size"],
                             nb_langevin=rp["pula_nb_langevin"], **common)
    torch.manual_seed(seed)
    out["P-DAPS"] = T.run_pdaps_v3(ode_steps=rp["pdaps_ode_steps"], **common)
    gt_x2 = T.sample_ground_truth(120_000)[:, 1]
    return {m: out[m]["final"][:, 1] for m in METHODS}, gt_x2


def _multiseed_metrics(scenario, seeds):
    """Per-method (W1_mean, W1_std, top_mean) and the gt top-mass, over `seeds`."""
    w1 = {m: [] for m in METHODS}
    up = {m: [] for m in METHODS}
    gt_up = []
    for s in seeds:
        finals, gt_x2 = _finals_for_seed(scenario, s)
        gt_up.append(_upper_mass(gt_x2))
        for m in METHODS:
            w1[m].append(_w1(finals[m], gt_x2))
            up[m].append(_upper_mass(finals[m]))
    res = {m: (float(np.mean(w1[m])), float(np.std(w1[m])), float(np.mean(up[m])))
           for m in METHODS}
    return res, float(np.mean(gt_up))


def fig_precond(data, out_dir):
    import matplotlib.pyplot as plt
    nrow = len(SCENARIOS)
    fig, axes = plt.subplots(nrow, 4, figsize=(11.0, 2.9 * nrow + 0.4),
                             sharex="row", constrained_layout=True)
    axes = np.atleast_2d(axes)
    order = ISO + PRE
    for r, scen in enumerate(SCENARIOS):
        out, geom, _ = data[scen]
        lim = geom["lim"]
        bins = np.linspace(-lim, lim, 70)
        gt_x2 = geom["gt"][:, 1]
        gt_h, _ = np.histogram(gt_x2, bins=bins, density=True)
        ctr = 0.5 * (bins[:-1] + bins[1:])
        ymax = gt_h.max() * 1.32
        ms, gt_up = _multiseed_metrics(scen, METRIC_SEEDS)
        for c, m in enumerate(order):
            ax = axes[r, c]
            x2 = out[m]["final"][:, 1]
            lab_p = "posterior (ground truth)" if (r == 0 and c == 0) else None
            lab_s = "sampler" if (r == 0 and c == 0) else None
            ax.fill_between(ctr, gt_h, color=L.DENSITY[2], alpha=0.9, lw=0,
                            zorder=1, label=lab_p)
            h, _ = np.histogram(x2, bins=bins, density=True)
            ax.plot(ctr, h, color=L.SAMPLE, lw=1.6, zorder=3, label=lab_s)
            for mode_y in np.unique(np.round(geom["modes"][:, 1], 3)):
                ax.axvline(mode_y, color=L.SUBTLE, lw=0.5, ls=":", zorder=0)
            w1_mean, w1_std, up_mean = ms[m]
            ax.text(0.5, 0.97,
                    rf"$W_1\!=\!{w1_mean:.3f}\!\pm\!{w1_std:.3f}$" + "\n"
                    + rf"top {up_mean:.2f} / {gt_up:.2f}",
                    transform=ax.transAxes, va="top", ha="center", fontsize=10.5,
                    color=L.INK, zorder=5)
            ax.set_ylim(0, ymax)
            ax.set_xlim(-lim, lim)
            ax.tick_params(labelsize=11, length=3)
            ax.set_yticks([])
            grp = "isotropic" if m in ISO else "preconditioned"
            if r == 0:
                ax.set_title(rf"{m}" + "\n" + rf"{{\small ({grp})}}", fontsize=15)
            if r == nrow - 1:
                ax.set_xlabel("$x_2$  (unmeasured axis)", fontsize=12)
            if c == 0:
                ax.set_ylabel(SCEN_LABEL[scen], fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=11)
    _save(fig, out_dir, "fig_toy_precond")
def _metrics(res, geom):
    steps = sorted(res["progress"])
    A = torch.tensor(geom["A"]); y = torch.tensor(geom["y"])
    sig, misfit, x2cov = [], [], []
    rp_sig = None
    for s in steps:
        X = torch.tensor(res["progress"][s]["samples"])
        r = (X @ A.T) - y
        misfit.append(float((r ** 2).sum(-1).sqrt().mean()))
        x2cov.append(float(X[:, 1].std()))
    return np.array(steps), np.array(misfit), np.array(x2cov)


def fig_convergence(data, out_dir):
    import matplotlib.pyplot as plt
    nrow = len(SCENARIOS)
    fig, axes = plt.subplots(nrow, 2, figsize=(9.5, 3.1 * nrow + 0.3),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    PRECOND = "#2e7d5b"
    styles = {"DPS": ("-", L.SAMPLE), "DAPS": ("--", L.SAMPLE),
              "pULA": ("-", PRECOND), "P-DAPS": ("--", PRECOND)}
    for r, scen in enumerate(SCENARIOS):
        out, geom, N = data[scen]
        rp = T.SCENARIOS[scen]["run_params"]
        sigmas_all = np.geomspace(rp["sigma_max"], rp["sigma_min"], N + 1)
        gt_x2_std = float(geom["gt"][:, 1].std())
        for m in METHODS:
            steps, misfit, x2cov = _metrics(out[m], geom)
            sig = sigmas_all[np.array(steps)]
            ls, col = styles[m]
            axes[r, 0].plot(sig, misfit, ls, color=col, lw=1.5, label=m)
            axes[r, 1].plot(sig, x2cov, ls, color=col, lw=1.5, label=m)
        axes[r, 1].axhline(gt_x2_std, color=L.SUBTLE, lw=0.9, ls=":",
                           label="posterior", zorder=0)
        for c, (ttl, log_y) in enumerate([("data misfit $\\|A x - y\\|$", True),
                                          ("$x_2$ spread (std)", False)]):
            ax = axes[r, c]
            ax.set_xscale("log"); ax.invert_xaxis()
            if log_y:
                ax.set_yscale("log")
            ax.tick_params(labelsize=11, length=3)
            if r == 0:
                ax.set_title(ttl, fontsize=14)
            if r == nrow - 1:
                ax.set_xlabel(r"noise scale $\sigma_t$  (cooling $\rightarrow$)",
                              fontsize=12)
        axes[r, 0].set_ylabel(SCEN_LABEL[scen], fontsize=12)
    axes[0, 0].legend(frameon=False, fontsize=10, ncol=2)
    _save(fig, out_dir, "fig_toy_convergence")


def _save(fig, out_dir, name):
    import matplotlib.pyplot as plt
    out = os.path.join(out_dir, name)
    fig.savefig(f"{out}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    ap.add_argument("--only", choices=["clouds", "precond", "convergence", "problem"],
                    default=None)
    args = ap.parse_args()
    L.apply_style()
    if args.only == "problem":
        fig_problem(args.out_dir)
        return
    data = {scen: run_all(scen) for scen in SCENARIOS}
    if args.only in (None, "clouds"):
        fig_clouds(data, args.out_dir)
    if args.only in (None, "precond"):
        fig_precond(data, args.out_dir)
    if args.only in (None, "convergence"):
        fig_convergence(data, args.out_dir)
    if args.only is None:
        fig_problem(args.out_dir)


if __name__ == "__main__":
    main()
