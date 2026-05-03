#!/usr/bin/env python3
"""Render 3D posterior surfaces with per-particle sample *traces*.

For each scenario we run one or more algorithms with dense per-step logging,
then draw a handful of individual particle trajectories on the log-posterior
surface so you can see where each algorithm walks as sigma is annealed.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import toy_2d as toy


METHOD_COLORS = {
    "DPS":         "#1b9e77",
    "DAPS":        "#d95f02",
    "pULA":        "#7570b3",
    "P-DAPS":      "#e7298a",
    "P-DAPS-warm": "#e6ab02",
}


def posterior_surface(grid_size, z_transform="sqrt"):
    """Posterior density on a 2D grid plus a callable that returns matching z
    for arbitrary (x1, x2) points. z_transform compresses dynamic range so
    particle traces remain visible against sharp posterior peaks.
      'none' -> raw density
      'sqrt' -> sqrt(density)
      'log1p' -> log1p(density * k) for a scale k picked from the grid.
    """
    xx, yy, grid_points = toy.make_grid(grid_size)
    logp_grid = toy.posterior_log_prob(grid_points).detach().cpu().numpy()
    logp_max = float(logp_grid.max())
    density_grid = np.exp(logp_grid - logp_max).reshape(xx.shape)
    total = float(max(density_grid.sum(), 1e-12))
    density_grid = density_grid / total

    if z_transform == "sqrt":
        transform = np.sqrt
    elif z_transform == "log1p":
        scale = 1.0 / max(float(density_grid.max()), 1e-12)
        transform = lambda d: np.log1p(d * scale)
    else:
        transform = lambda d: d

    z_grid = transform(density_grid)

    def z_at(points):
        pts = torch.as_tensor(points, dtype=toy.DTYPE, device=toy.DEVICE)
        logp = toy.posterior_log_prob(pts).detach().cpu().numpy()
        d = np.exp(logp - logp_max) / total
        return transform(d)

    return xx, yy, z_grid, z_at


def run_methods_with_traces(scenario, nb, N_override=None):
    toy.configure_scenario(scenario, quiet=True)
    rp = toy.CURRENT_SCENARIO["run_params"]
    N = N_override or rp["N"]
    budgets = tuple(range(1, N + 1))

    results = {
        "DAPS": toy.run_daps(
            nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["daps_ode_steps"], lgvd_steps=rp["daps_langevin_steps"],
            lgvd_lr=rp["daps_langevin_lr"], budget_steps=budgets,
        ),
        "pULA": toy.run_pula(
            nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            step_size=rp["pula_step_size"], nb_langevin=rp["pula_nb_langevin"],
            budget_steps=budgets,
        ),
        "P-DAPS-warm": toy.run_pdaps_warm(
            nb=nb, N=N, sigma_max=rp["sigma_max"], sigma_min=rp["sigma_min"],
            ode_steps=rp["pdaps_ode_steps"], lgvd_steps=rp["pdaps_langevin_steps"],
            lgvd_step_size=rp["pdaps_langevin_step_size"],
            warm_fraction=rp["pdaps_warm_fraction"], budget_steps=budgets,
        ),
    }
    return results


def extract_traces(result, particle_indices):
    """Return (T, K, 2) array of tracked particles over recorded steps."""
    steps = sorted(result["progress"].keys())
    if not steps:
        return np.empty((0, 0, 2)), steps
    stacked = np.stack([result["progress"][s]["samples"] for s in steps], axis=0)
    traces = stacked[:, particle_indices, :]
    return traces, steps


def pick_particle_indices(n_available, n_tracks, rng):
    n_tracks = min(n_tracks, n_available)
    return rng.choice(n_available, size=n_tracks, replace=False)


def plot_scenario(scenario, results, n_tracks, grid_size, out_path, seed=0,
                  z_transform="sqrt"):
    xx, yy, density_grid, density_at = posterior_surface(grid_size, z_transform=z_transform)
    rng = np.random.default_rng(seed)

    methods = list(results.keys())
    fig = plt.figure(figsize=(5.8 * len(methods), 6.2))

    for idx, method in enumerate(methods, start=1):
        result = results[method]
        ax = fig.add_subplot(1, len(methods), idx, projection="3d")
        ax.plot_surface(xx, yy, density_grid, cmap="viridis",
                        linewidth=0, antialiased=True, alpha=0.62,
                        edgecolor="none", rstride=1, cstride=1, shade=True)

        final = result["final"]
        nb = final.shape[0]
        particle_idx = pick_particle_indices(nb, n_tracks, rng)
        traces, steps = extract_traces(result, particle_idx)
        if traces.size == 0:
            ax.set_title(f"{method} (no progress recorded)")
            continue

        # Clamp traces to view so density lookup stays valid.
        clipped = np.clip(traces, -toy.LIM, toy.LIM)
        color = METHOD_COLORS.get(method, "#333333")

        # Lift the trace slightly above the surface so it doesn't z-fight.
        z_lift = 0.015 * float(np.nanmax(density_grid))

        for k in range(clipped.shape[1]):
            xy = clipped[:, k, :]
            z = density_at(xy) + z_lift
            T = xy.shape[0]
            # Color fades from light to the method's base color along time.
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            zs = np.stack([z[:-1], z[1:]], axis=1)
            for seg_i in range(T - 1):
                frac = 0.35 + 0.65 * (seg_i / max(T - 1, 1))
                ax.plot(segments[seg_i, :, 0], segments[seg_i, :, 1], zs[seg_i],
                        color=color, lw=1.0, alpha=frac, solid_capstyle="round")
            ax.scatter(xy[0, 0], xy[0, 1], z[0], color="white", marker="o",
                       s=22, edgecolors=color, linewidths=1.2, zorder=5)
            ax.scatter(xy[-1, 0], xy[-1, 1], z[-1], color=color, marker="^",
                       s=42, edgecolors="black", linewidths=0.6, zorder=6)

        diverged = result.get("diverged_at")
        subtitle = f"{method}"
        if diverged is not None:
            subtitle += f" (diverged at step {diverged})"
        subtitle += f"\n{len(particle_idx)} traces over {len(steps)} outer steps"
        ax.set_title(subtitle, pad=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel(f"density ({z_transform})")
        ax.set_xlim(-toy.LIM, toy.LIM)
        ax.set_ylim(-toy.LIM, toy.LIM)
        ax.view_init(elev=32, azim=-58)

    fig.suptitle(f"{toy.CURRENT_SCENARIO['name']}: particle traces on posterior surface",
                 y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+",
                   default=["toy_a_mode_recovery", "toy_b_stiffness", "toy_c_score_bias"])
    p.add_argument("--nb", type=int, default=300,
                   help="Number of particles per method (only n-tracks are drawn)")
    p.add_argument("--n-tracks", type=int, default=10,
                   help="Number of particle trajectories to draw per method")
    p.add_argument("--grid-size", type=int, default=140)
    p.add_argument("--N-override", type=int, default=None,
                   help="Override outer-step count (None = scenario default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="imgs/3d_traces")
    p.add_argument("--z-transform", choices=("none", "sqrt", "log1p"), default="sqrt",
                   help="Compress z-axis so traces remain visible next to sharp peaks")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for scenario in args.scenarios:
        print(f"Rendering traces for {scenario}")
        toy.set_global_seed(args.seed)
        results = run_methods_with_traces(scenario, nb=args.nb, N_override=args.N_override)
        out_path = os.path.join(args.out_dir, f"{scenario}_traces_3d.png")
        plot_scenario(scenario, results, n_tracks=args.n_tracks,
                      grid_size=args.grid_size, out_path=out_path, seed=args.seed,
                      z_transform=args.z_transform)
    print(f"Done. Figures in {args.out_dir}")


if __name__ == "__main__":
    main()
