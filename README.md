# P-DAPS

Minimal public implementation of P-DAPS for Cartesian multicoil MRI posterior sampling.

This repository is intentionally small. It contains the P-DAPS sampler, the pULA baseline used for comparison, and shared Cartesian multicoil MRI linear algebra helpers. It uses the project fork of [InverseBench](https://github.com/dedelstein/InverseBench) as a source dependency for the base algorithm interface, scheduler, diffusion sampler, models, and inverse-problem utilities.

The accompanying thesis is included as [`Dan_Edelstein_s243446_MSc_Thesis.pdf`](Dan_Edelstein_s243446_MSc_Thesis.pdf).

## Layout

```text
mri_ops.py            # conjugate gradient and Cartesian multicoil MRI operators
pula.py               # pULA baseline implementation
pdaps.py              # P-DAPS sampler, formerly pdaps_v3
metrics.py            # compact MRI reconstruction metrics
dataloader.py         # fastMRI multicoil HDF5 + ESPIRiT-map dataset
espirit.py            # ESPIRiT sensitivity-map precomputation utility
mri_validation.py     # minimal single-slice MRI runner
toy_2d.py             # synthetic 2-D toys plus Toy D, the 64-D stiffness toy
thesis_figures/       # data-free figure-generation scripts; no generated PNGs
libs/inversebench/    # InverseBench submodule
Dan_Edelstein_s243446_MSc_Thesis.pdf
```

## Install

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/dedelstein/pdaps.git
cd pdaps
pip install -e libs/inversebench
pip install -e .
```

If the repository was already cloned:

```bash
git submodule update --init --recursive
```

## Interface

`PDAPS` expects:

- `net`: an InverseBench-style denoiser with `img_channels`, `img_resolution`, and `forward(x, sigma)`.
- `forward_op`: a Cartesian multicoil MRI operator with `device`, `maps`, `mask`, `fft`, and `ifft`.
- `observation`: real-view complex k-space with shape compatible with `torch.view_as_complex`.

Basic use:

```python
from pdaps import PDAPS
from pula import PULA

sampler = PDAPS(
    net=net,
    forward_op=forward_op,
    annealing_scheduler_config={"num_steps": 10, "sigma_max": 80, "sigma_min": 0.01},
    diffusion_scheduler_config={"num_steps": 10, "sigma_min": 0.01},
)

recon = sampler.inference(observation, num_samples=1)
```

Metrics:

```python
from metrics import compute_metrics

scores = compute_metrics(forward_op, recon, target, observation)
```

Posterior diagnostics for multiple samples:

```python
from metrics import posterior_metrics

samples = [sampler.inference(observation, num_samples=1) for _ in range(8)]
summary = posterior_metrics(forward_op, samples, target, observation)
```

Minimal MRI run:

```bash
python mri_validation.py \
  --models-dir /path/to/models \
  --ckpt-name MRI-knee.pt \
  --kspace-dir /path/to/fastmri/multicoil_val \
  --maps-dir /path/to/espirit_maps \
  --filename file1000196.h5 \
  --slice-rank 0 \
  --method pdaps \
  --num-samples 1
```

ESPIRiT maps:

```bash
python espirit.py --data /path/to/fastmri/multicoil_val --out-dir /path/to/espirit_maps
```

`espirit.py` requires a CuPy build matching your CUDA runtime. The core sampler
modules do not import CuPy.

fastMRI data and tools:

- Official fastMRI code repository: https://github.com/facebookresearch/fastMRI
- Official fastMRI dataset page: https://fastmri.med.nyu.edu/

## Notes

This public version does not include thesis experiment scripts, local cluster jobs, private data paths, checkpoints, generated results, or validation artifacts.
