r"""Inner temperature audit: injected noise is neutral at best (Sec. dev_noise).

The nominal pDAPS contribution is a preconditioned Langevin sampler of the inner
posterior. We audited the temperature directly against its zero-temperature limit
(the deterministic correction). On the unmeasured subspace the data term vanishes,
so the only restoring force is the tau^2-attenuated anchor (rate ~ beta_y^2 ~ 4e-6):
full CN(0, M_t) noise drives a nearly undamped random walk and quality collapses to
single-digit PSNR. Restricting noise to the measured subspace removes the blow-up
but does not help, declining gently below the drift baseline. The zero-temperature
drift limit matches or beats every stochastic configuration.

Data (verified, single validation slice n=1, gamma=0.5, R=4):
  results/mri_validation_pdaps_v8a_accel4_28494260/validation_summary.csv
    drift (tau=0)   : 31.63 dB         (v8a_drift_g0p5)
    full      noise : 7.38 / 4.52 / 2.80 at tau 0.025 / 0.05 / 0.1
    range_only noise: 31.55/31.47/31.29/30.88 at tau 0.025/0.05/0.1/0.2

Run:  ./.venv/bin/python3 thesis_figures/fig_dev_noise_collapse.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import landscape as L

DRIFT = 31.63
FULL_T = [0.025, 0.05, 0.1]
FULL_P = [7.38, 4.52, 2.80]
RANGE_T = [0.025, 0.05, 0.1]
RANGE_P = [31.55, 31.47, 31.29]


def main(out: Path) -> None:
    L.apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axhline(DRIFT, color=L.SUBTLE, lw=1.4, ls=":", zorder=1)
    ax.annotate(r"drift, $\mathrm{temp}=0$ (" + f"{DRIFT:.1f} dB)", (0.0, DRIFT),
                xytext=(2, -16), textcoords="offset points", color=L.SUBTLE, fontsize=11.5)
    ax.plot([0] + RANGE_T, [DRIFT] + RANGE_P, color=L.SAMPLE, lw=2.0,
            marker="s", ms=7, ls="--", zorder=5)
    ax.annotate("range-only noise", (RANGE_T[1], RANGE_P[1]), xytext=(-4, 9),
                textcoords="offset points", color=L.SAMPLE, fontsize=12.5, ha="center")
    ax.plot([0] + FULL_T, [DRIFT] + FULL_P, color=L.SAMPLE, lw=2.0,
            marker="o", ms=8, ls="-", zorder=6)
    ax.annotate(r"full $\mathcal{CN}(0,\mathbf{M}_t)$ noise", (FULL_T[1], FULL_P[1]),
                xytext=(10, 6), textcoords="offset points", color=L.INK, fontsize=12.5)

    ax.set_yscale("log")
    ax.set_xlabel(r"inner noise temperature", fontsize=15)
    ax.set_ylabel("validation PSNR (dB, log scale)", fontsize=15)
    ax.set_xlim(-0.004, 0.114)
    ax.set_ylim(2.2, 46)
    ax.set_xticks([0, 0.025, 0.05, 0.1])
    ax.set_xticklabels(["0", "0.025", "0.05", "0.1"])
    ax.set_yticks([2, 5, 10, 20, 31])
    ax.set_yticklabels(["2", "5", "10", "20", "31"])
    ax.text(0.0, 1.04, r"Full inner noise collapses the reconstruction; "
            r"measured-subspace noise only decays below the deterministic drift.",
            transform=ax.transAxes, fontsize=12.3, color=L.INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "fig_dev_noise_collapse.png")
