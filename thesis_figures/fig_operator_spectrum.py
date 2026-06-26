"""The spectrum of the accelerated MRI operator (Bottleneck and CG sections).

A 1-D multi-coil analogue of A = P F S (n voxels, N_c coils with smooth
overlapping sensitivities normalised to sum_c |s_c|^2 = 1, unitary DFT,
equispaced undersampling + ACS), small enough to eigendecompose exactly.

  * spectrum  : the mask alone has the clean {0,1} spectrum; composing with the
                coil sensitivities S leaves a band of small nonzero eigenvalues
                ahead of the exact null space -- kappa+ is read off the plot;
  * nu-shift  : A^H A + nu I lifts the whole spectrum by nu; the null space
                becomes the nu-eigenspace and kappa(B) = 1 + lambda_max/nu
                is finite;
  * iterations: gradient descent vs conjugate gradient on the regularised
                normal equations -- the O(kappa) vs O(sqrt(kappa)) gap.

Run:  ./.venv/bin/python3 thesis_figures/fig_operator_spectrum.py
"""

from __future__ import annotations

import argparse
import numpy as np

import landscape as L

N = 192               # voxels
NC = 4                # coils
R = 8                 # acceleration
N_ACS = 16            # fully sampled centre lines
NU = 1e-2             # regularisation (Tikhonov / Gaussian-prior precision)
TOL = 1e-10           # numerical-rank threshold (relative to lambda_max)


def build_operator():
    r = np.arange(N)
    centers = np.linspace(0.1, 0.9, NC) * N
    width = 0.35 * N
    mags = np.exp(-0.5 * ((r[None, :] - centers[:, None]) / width) ** 2)
    phases = np.exp(1j * 2 * np.pi * (0.3 * np.sin(2 * np.pi * r / N)[None, :]
                                      * np.linspace(-1, 1, NC)[:, None]))
    s = mags * phases
    s /= np.sqrt((np.abs(s) ** 2).sum(0, keepdims=True))     # sum_c |s_c|^2 = 1
    F = np.exp(-2j * np.pi * np.outer(r, r) / N) / np.sqrt(N)
    keep = np.zeros(N, bool)
    keep[::R] = True
    keep[N // 2 - N_ACS // 2: N // 2 + N_ACS // 2] = True
    Fsel = F[keep]
    A = np.vstack([Fsel @ np.diag(s[c]) for c in range(NC)])
    A0 = Fsel                                                # mask-only (s == 1)
    return A, A0


def eigvals(A):
    lam = np.linalg.eigvalsh(A.conj().T @ A).real[::-1]
    return np.clip(lam, 0.0, None)


def run_gd_cg(A, x_true, iters_gd, iters_cg):
    """Solve (A^H A + nu I) x = A^H y; return error norms per iteration."""
    y = A @ x_true
    b = A.conj().T @ y
    B = A.conj().T @ A + NU * np.eye(N)
    x_star = np.linalg.solve(B, b)
    eta = 1.0 / np.linalg.eigvalsh(B).real.max()

    x = np.zeros(N, complex)
    gd = [np.linalg.norm(x - x_star)]
    for _ in range(iters_gd):
        x = x - eta * (B @ x - b)
        gd.append(np.linalg.norm(x - x_star))

    x = np.zeros(N, complex)
    rvec = b - B @ x
    p = rvec.copy()
    cg = [np.linalg.norm(x - x_star)]
    for _ in range(iters_cg):
        Bp = B @ p
        alpha = (rvec.conj() @ rvec).real / (p.conj() @ Bp).real
        x = x + alpha * p
        r_new = rvec - alpha * Bp
        beta = (r_new.conj() @ r_new).real / (rvec.conj() @ rvec).real
        p = r_new + beta * p
        rvec = r_new
        cg.append(np.linalg.norm(x - x_star))
    return np.array(gd), np.array(cg)


def main():
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="thesis_figures")
    args = ap.parse_args()

    L.apply_style()
    A, A0 = build_operator()
    lam, lam0 = eigvals(A), eigvals(A0)
    lmax = lam[0]
    pos = lam > TOL * lmax
    lmin_pos = lam[pos][-1]
    kappa_plus = lmax / lmin_pos
    kappa_b = 1.0 + lmax / NU
    idx = np.arange(1, N + 1)
    floor = 10 ** np.floor(np.log10(TOL * lmax)) * 0.3

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(10.0, 3.4), gridspec_kw=dict(wspace=0.32))
    ax1.semilogy(idx, np.clip(lam0, floor, None), color=L.SUBTLE, lw=1.0,
                 ls="--", drawstyle="steps-post")
    ax1.semilogy(idx[pos], lam[pos], color=L.SAMPLE, lw=1.3)
    ax1.semilogy(idx[~pos], np.full((~pos).sum(), floor), color=L.SAMPLE,
                 lw=1.3, ls=":")
    ax1.text(0.97, 0.92, r"mask alone ($\mathcal{PF}$): $\{0,1\}$",
             transform=ax1.transAxes, fontsize=13.5, color=L.INK, ha="right")
    ax1.text(0.04, 0.62, r"$\mathbf{A}=\mathcal{PFS}$", fontsize=13.5,
             color=L.SAMPLE, transform=ax1.transAxes)
    ax1.annotate(r"band of small $\lambda_i$",
                 xy=(idx[pos][-8], lam[pos][-8]), xytext=(0.38, 0.36),
                 textcoords="axes fraction", fontsize=13, color=L.INK,
                 arrowprops=dict(arrowstyle="-", color=L.SUBTLE, lw=0.6))
    ax1.text(0.97, 0.10, r"$\operatorname{null}(\mathbf{A})$: $\lambda=0$",
             transform=ax1.transAxes, fontsize=13, color=L.SAMPLE, ha="right")
    ax1.text(0.04, 0.50, rf"$\kappa^+ = \frac{{\lambda_{{\max}}}}{{\lambda_{{\min}}^+}}"
             rf"\approx 10^{{{np.log10(kappa_plus):.0f}}}$",
             transform=ax1.transAxes, fontsize=13.5, color=L.INK)
    ax1.set_ylabel(r"eigenvalue  $\lambda_i$", fontsize=15)
    ax1.set_title(r"eigenvalues of $\mathbf{A}^H\mathbf{A}$", fontsize=16,
                  loc="left")
    ax2.semilogy(idx, np.clip(lam, floor, None), color=L.SAMPLE, lw=1.0,
                 alpha=0.45)
    ax2.semilogy(idx, lam + NU, color=L.SAMPLE, lw=1.3)
    ax2.axhline(NU, color=L.TRUTH, lw=0.9, ls="--")
    ax2.text(0.04, 0.40, r"$\nu$", fontsize=14.5, color=L.TRUTH,
             transform=ax2.transAxes)
    ax2.text(0.04, 0.86, r"$\lambda_i + \nu$", fontsize=13.5, color=L.SAMPLE,
             transform=ax2.transAxes)
    ax2.text(0.04, 0.74, r"$\lambda_i$", fontsize=13.5, color=L.SAMPLE,
             alpha=0.55, transform=ax2.transAxes)
    ax2.text(0.97, 0.10, r"$\kappa(\mathbf{B}) = 1+\frac{\lambda_{\max}}{\nu}$"
             "\n(finite)", transform=ax2.transAxes, fontsize=13.5, color=L.INK,
             ha="right")
    ax2.set_ylabel(r"eigenvalue", fontsize=15)
    ax2.set_title(r"$\mathbf{B}=\mathbf{A}^H\mathbf{A}+\nu\mathbf{I}$",
                  fontsize=16, loc="left")
    rng = np.random.default_rng(0)
    rr = np.arange(N)
    x_true = (np.exp(-0.5 * ((rr - 70) / 14.0) ** 2)
              + 0.7 * np.exp(-0.5 * ((rr - 130) / 9.0) ** 2)).astype(complex)
    x_true += 0.02 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    gd, cg = run_gd_cg(A, x_true, iters_gd=400, iters_cg=60)
    e0 = gd[0]
    ax3.semilogy(np.arange(len(gd)), gd / e0, color=L.SUBTLE, lw=1.2)
    ax3.semilogy(np.arange(len(cg)), cg / e0, color=L.SAMPLE, lw=1.3)
    ax3.text(0.55, 0.80, r"gradient descent: $\mathcal{O}(\kappa)$",
             fontsize=13.5, color=L.INK, transform=ax3.transAxes)
    ax3.text(0.18, 0.30, r"conjugate gradient: $\mathcal{O}(\sqrt{\kappa})$",
             fontsize=13.5, color=L.SAMPLE, transform=ax3.transAxes)
    ax3.set_xlabel("iteration $k$", fontsize=14.5)
    ax3.set_ylabel(r"relative error  $\|\mathbf{x}_k-\mathbf{x}_\nu\|$",
                   fontsize=15)
    ax3.set_title(r"solving $\mathbf{B}\mathbf{x}=\mathbf{A}^H\mathbf{y}$",
                  fontsize=16, loc="left")

    for ax in (ax1, ax2):
        ax.set_xlabel("eigenvalue index $i$", fontsize=14.5)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=13, length=3)
    fig.text(0.012, 0.905, rf"1-D multi-coil analogue of $\mathbf{{A}}="
             rf"\mathcal{{PFS}}$ ($n={N}$ voxels, $N_c={NC}$ coils, $R={R}$ "
             rf"equispaced + {N_ACS}-line ACS, $\nu=10^{{{int(np.log10(NU))}}}$)",
             fontsize=13.5, color=L.INK)
    fig.subplots_adjust(top=0.74, bottom=0.16, left=0.06, right=0.99)

    out = f"{args.out_dir}/fig_operator_spectrum"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    print(f"wrote {out}.png  "
          f"(rank={pos.sum()}, null dim={N - pos.sum()}, "
          f"kappa+={kappa_plus:.2e}, kappa(B)={kappa_b:.1f})")


if __name__ == "__main__":
    main()
