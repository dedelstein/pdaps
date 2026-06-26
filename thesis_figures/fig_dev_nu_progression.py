r"""pDAPS development progression: the two inner-regularization fixes (Sec. dev_nu).

The early iterative pDAPS started from the unweighted normal equations with the
prior precision nu = 1/sigma_t^2 (pDAPS_0). Two corrections, applied in order,
lifted it above DAPS for the first time:

  1. tau^2 rescale: nu = beta_y^2 / sigma_t^2 aligns the prior:likelihood ratio
     with the objective DAPS actually tunes (the anchor was ~2.4e5 too strong).
  2. gamma ~ 1/sigma_t^2 step schedule: spends the inner budget at low sigma_t
     where x0_hat is informative, instead of stalling under a fixed step.

The gamma-cap variants (lambda_cap, sqrt_lambda_cap) clip exactly the sigma<1
regime that does the work, reverting to the pre-schedule level: the cap erases
the second gain. Shown as a hollow marker.

Data (verified, single validation slice, n=1):
  results/mri_validation_pdaps_v5_accel4_28429098  (R=4)
  results/mri_validation_pdaps_v5_accel4_28429106  (R=8, dir mislabelled accel4)
    pDAPS_0  = v5a_tau1_unfloored
    +tau^2   = v5a_taufix
    +gamma   = v5b_taufix_glambda
    cap      = v5b_taufix_glambdacap  (== taufix level)
    DAPS     = DAPS reference row

Run:  ./.venv/bin/python3 thesis_figures/fig_dev_nu_progression.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import landscape as L
STAGES = [r"$\mathrm{pDAPS}_0$", r"$+\,\tau^2$ rescale", r"$+\,\gamma\propto1/\sigma_t^2$"]
PSNR = {
    4: [28.44, 28.98, 30.83],
    8: [27.40, 28.21, 29.44],
}
DAPS = {4: 29.45, 8: 26.85}
CAP = {4: 28.98, 8: 28.21}   # gamma-cap reverts the third stage to the taufix level

STYLE = {4: dict(ls="-", marker="o"), 8: dict(ls="--", marker="s")}


def main(out: Path) -> None:
    L.apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = list(range(len(STAGES)))

    for R in (4, 8):
        ax.plot(x, PSNR[R], color=L.SAMPLE, lw=2.0, ms=8, **STYLE[R], zorder=5)
        ax.annotate(rf"$R={R}$", (x[-1], PSNR[R][-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=L.SAMPLE, fontsize=13)
        ax.axhline(DAPS[R], color=L.SUBTLE, lw=1.2, ls=":", zorder=1)
        ax.annotate(rf"DAPS $R={R}$", (0, DAPS[R]), xytext=(2, 4 if R == 4 else -14),
                    textcoords="offset points", color=L.SUBTLE, fontsize=11.5)
        ax.plot([x[-1]], [CAP[R]], marker="o", mfc="white", mec=L.SAMPLE,
                ms=9, mew=1.6, ls="none", zorder=6)
        ax.plot([x[-1], x[-1]], [CAP[R], PSNR[R][-1]], color=L.SAMPLE, lw=1.0,
                ls=(0, (1, 1)), alpha=0.6, zorder=2)

    ax.annotate(r"$\gamma$-cap reverts", (x[-1], CAP[4]), xytext=(-12, -20),
                textcoords="offset points", ha="right", color=L.INK, fontsize=11.5,
                arrowprops=dict(arrowstyle="->", color=L.INK, lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=14)
    ax.set_ylabel("validation PSNR (dB)", fontsize=15)
    ax.set_xlim(-0.25, len(STAGES) - 1 + 0.55)
    ax.margins(y=0.12)
    ax.text(0.0, 1.04, r"Each inner-regularization fix lifts pDAPS, the second past DAPS; "
            r"the $\gamma$-cap erases it.",
            transform=ax.transAxes, fontsize=13, color=L.INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "fig_dev_nu_progression.png")
