# Thesis Figures

This folder contains figure-generation scripts that do not require private MRI
data, generated PNG assets, saved reconstruction tensors, run logs, or NPZ
trajectory dumps.

The shared visual language lives in `landscape.py`. The synthetic toy figures
use `../toy_2d.py`, which includes the 2-D scenarios and Toy D, the 64-dimensional
stiff linear-Gaussian-GMM scenario.

Run from the repository root, for example:

```bash
python thesis_figures/fig_sampler_structure.py --out-dir thesis_figures
python thesis_figures/fig_toy_samplers.py --only clouds --out-dir thesis_figures
```

Generated PNG/PDF outputs are intentionally ignored by Git.

