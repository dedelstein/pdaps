# Thesis figure design & layout philosophy

The conventions and hard-won lessons behind `thesis_figures/`. Read this before
adding or restyling a figure. Everything flows from `landscape.py::apply_style`
(shared rcParams, palette, usetex preamble) — change it there, not per-file.

---

## 1. Design philosophy

**One visual language across the whole methods narrative (Ch. 2–4).** A reader
learns the vocabulary once and reads every figure:

| glyph / colour | meaning |
| --- | --- |
| orange `×` (`TRUTH`) | the true signal / solution / minimiser `$x^\star$` |
| blue (`SAMPLE` `#1f4f9e`) | sampler output: walkers, chains, sample clouds, paths |
| gray tints (`DENSITY`) | a probability density — **darker = larger** |
| open `○` | a start point, or a posterior mode |
| `SUBTLE` gray | faint *reference lines only* (envelopes, dotted axes) — **never text** |

**Characterize, don't stage flaws.** The 2-D figures show *how* SOTA samplers
differ in mechanism (preconditioning, decoupling, annealing), never a rigged
"our method wins." Honesty constraints we hit:
- *Decoupling's benefit is high-dimensional* — it does **not** show in 2D, so
  `fig_decoupling`/`fig_sampler_structure` show update *structure/character*, not
  a coverage contest. Don't draw "decoupled escapes a mode, coupled stuck."
- *Preconditioning is the dimension-independent axis* — it does show cleanly.
- Quantile-shaded density bands are *value* quantiles, not credible-mass regions.

**Above all, show the data.** Maximize data-ink; titles → captions (see §2);
direct in-panel labels over distant legends where a single colour suffices;
small multiples for the comparative/annealed stories.

**Match the thesis's analytic notation.** `\sigma_t` not `\sigma(t)`; `\frac{D}{\lambda}`
not `D/\lambda`; `\mathbb{E}[x_0\mid x]` (posterior mean) for the Tweedie target;
start `x_0`, iterates `x_k`, minimiser `x^\star`. Don't invent symbols the paper
doesn't use. For the multi-coil MRI forward model use the thesis symbols: bold
`\mathbf{A}=\mathcal{P}\mathcal{F}\mathcal{S}` with undersampling mask `\mathcal{P}`,
Fourier `\mathcal{F}`, sensitivities `\mathcal{S}`; per-coil maps `s_c(\mathbf{r})`;
adjoint `\mathbf{A}^H`; bold states `\mathbf{x},\mathbf{y}`; `N_c` coils. Source the
canonical form from `full_tex.tex` (grep it) before putting any thesis symbol on a
figure.

---

## 2. Hard-won technical lessons

1. **usetex + Agg silently ignores inline `\textcolor`.** Each text artist is
   rendered as a single-colour alpha mask tinted by the artist's `color`. We
   verified: a `\textcolor`-laden legend string had **0 coloured pixels**.
   - A *whole* `ax.text(..., color=ORANGE)` is fine (one colour).
   - A *multi-colour legend* must be a real `fig.legend(handles=…)` with **proxy
     artists** (`Line2D`/`Patch`), which *are* drawn in colour. The glyph carries
     the colour, so labels drop "blue ="/"gray =" wording. Always pass
     `labelcolor=L.INK` (black labels) and `borderaxespad=0.0`.
   - Verify colour landed: count pixels where channel-spread > 40 and not near-white.

2. **Figures are shrunk to `\linewidth`, so a 10in canvas downscales ~0.6× → 8pt
   text renders ~5pt.** Two consequences:
   - Author fonts **big** (base 13–16pt; see §3).
   - **Set the figure width to the widest heading** so it isn't downscaled more
     than necessary, and so the panels fill that width. A heading wider than the
     figure *overflows* — `bbox_inches="tight"` then balloons the saved PNG (we
     hit a 14.8in image) and the panels look tiny/inset. Fix by widening
     `figsize` **or** splitting the heading to 2 lines (prefer 2 lines over an
     absurdly wide figure). Verify with the pixel-extent check:
     ```python
     a = np.asarray(Image.open(png).convert("L")); h, w = a.shape
     def ext(y0, y1):
         b = a[int(h*y0):int(h*y1)]; c = np.where((b < 200).any(0))[0]
         return c.min()/w, c.max()/w        # heading extent should be <= panel extent
     ```

3. **Title lives in the code; the subheading lives on the image.** Two different
   registers, and they must not be swapped:
   - **Title** — *broadly* announces the figure's topic (reads like the opening of
     the LaTeX caption). It is **never rendered**: write it as a plain comment,
     `# title (-> caption): ...`, purely so the caption is easy to write later.
     Never pass it to `fig.suptitle`/`ax.set_title`/`fig.text`.
   - **Subheading** — *analytically* describes what the panels show, in
     operator/mechanism terms (one capitalised sentence). This, plus the legend,
     is the **only** heading text that enters the image — nothing else above the
     panels.

   Rule of thumb: if a line broadly discusses the topic, it is the title
   (→ comment); if it analytically describes the mechanism, it is the subheading
   (→ image). Example: *Accelerated acquisition...* broadly announces → title
   (comment); *The operator $\mathbf{A}=\mathcal{P}\mathcal{F}\mathcal{S}$, shown
   single-coil...* analytically describes → subheading (on image).

4. **Bigger fonts overflow narrow panels.** Long *equation* panel-titles drop to
   ~12pt; secondary in-panel annotations to ~10.5–11pt; keep panel titles a
   **consistent size within a figure**. If multi-panel titles still collide,
   shorten them to acronyms (ODE/SDE/PDE) — the subheading carries the prose.

5. **Non-uniform horizontal gaps need a spacer column.** `GridSpec` `wspace` is
   uniform; to separate two panel *groups* (e.g. VE-margin | VP-margin) without
   over-spacing each panel-and-its-margin, insert a thin empty spacer column in
   `width_ratios`. (A too-wide spacer reads as a gulf — keep it ~0.10.)

---

## 3. Layout conventions (apply to every figure)

- **Heading block:** one capitalised subheading sentence, then the colour legend
  just below it. Keep the subheading to a single short sentence that fits the
  figure width; let the panel labels, arrows, and legend carry the per-symbol
  detail, so the sentence states the mechanism rather than re-listing symbols.
- **Normalized heading→legend gap:** legend `bbox_to_anchor` y ≈ **narrative_y −
  0.015**. Legend-only figures (no subheading): legend y ≈ 0.96.
- **Gap below the legend** to the graphs is slightly *larger* — tune
  `subplots_adjust(top=…)` / `tight_layout(rect=…)` so panel titles clear the
  legend. (2pt ≈ 0.008 fig-fraction at ~3.7in tall — small nudges matter.)
- **Legend:** every figure keys its colour/glyph encoding on the image — a real
  `fig.legend(handles=…)` of proxy artists for discrete glyphs, or a slim colour
  bar for a continuous/cyclic map (e.g. sensitivity magnitude or phase). A direct
  panel label that already names the encoding is itself the key. Mechanics:
  `fig.legend(handles=…, loc="upper left", bbox_to_anchor=(0.01, y), ncol=N,
  frameon=False, labelcolor=L.INK, borderaxespad=0.0)`. Keep it **one row**
  (`ncol` = item count) to avoid colliding with panel titles.
- **Fonts (authored):** subheading/legend 13–13.5; axis labels 14.5–16; panel
  titles 14–16 (12 for long equations); in-panel annotations 10.5–13.
- **All text black** (`L.INK`); only faint reference *lines* stay `SUBTLE` gray.
- **300 dpi** PNG, PNG-only (no PDFs).
- Add row gap in tall 2×2 figures with `tight_layout(..., h_pad=3.5)`.

---

## 4. Proxy-legend cookbook

```python
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Line2D([], [], marker="x", ls="none", color=L.TRUTH, ms=9, mew=1.8,
           label=r"solution $x^\star$"),                       # orange x
    Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=8,
           label="sample / estimate"),                         # solid blue dot
    Line2D([], [], marker="o", ls="none", mfc="white", mec=L.SAMPLE, ms=8,
           label=r"start $x_0$"),                              # open circle
    Line2D([], [], marker="o", ls="none", color=L.SAMPLE, ms=6, alpha=0.4,
           label="all chain endpoints"),                       # pale cloud
    Line2D([], [], color=L.SAMPLE, lw=1.6, label="a trajectory"),        # line
    Line2D([], [], color=L.TRUTH, lw=1.6, ls="--", label="reference"),   # dashed
    Patch(facecolor=L.DENSITY[2], edgecolor="none", label=r"density $p$"),  # gray
]
```

---

## 5. Figure inventory

Backported to this style: ode_pde_sde, wiener, langevin_overdamped,
kspace_sampling, ve_vp, diffusion, score_tweedie, gd_conditioning, gd_nullspace,
preconditioning, decoupling, sampler_structure, annealed_bridge, bowl_intuition,
ula_sampling, dhariwal_unet.

Not yet restyled (legend still a `fig.text`): **operator_spectrum**,
**sampler_trajectories**.

Imported / non-generated (leave as-is): daps_paper_*, pula_paper_*,
dps_paper_*, old_toy_2d_comparison.
