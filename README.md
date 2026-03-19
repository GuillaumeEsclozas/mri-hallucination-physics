# Physics-Informed Hallucination Detection in Accelerated MRI

Detecting and localizing hallucinated content in deep learning-based MRI reconstruction without ground truth access. Benchmarks 22 detectors across 6 families (physics-informed, uncertainty, learned, OOD, spectral, hybrid) on the [fastMRI](https://fastmri.med.nyu.edu/) single-coil knee dataset.

> **Best GT-free detector: Deep Ensemble at 0.859 patch-level AUROC**, with IFFT Gradient (0.828) as the best zero-cost alternative.

![Null-Space Decomposition](figures/figures_notebook_3/fig_001.png)

*IFFT vs U-Net null-space decomposition: IFFT error is 100% null-space (predictable from physics). U-Net error splits into hallucinated content and residual aliasing.*

## Key Results

| Detector | AUROC | GT-Free | Cost |
|---|---|---|---|
| Deep Ensemble (3 models) | 0.859 ± 0.081 | Yes | 3 forward passes |
| IFFT Gradient | 0.828 ± 0.065 | Yes | Negligible |
| MM + IFFT Grad (60/40) | 0.804 ± 0.078 | Yes | 9 forward passes |
| TTA Flip | 0.801 ± 0.085 | Yes | 2 forward passes |
| Multi-Mask (8 masks) | 0.776 ± 0.082 | Yes | 8 forward passes |
| MC Dropout (p=0.05) | 0.653 ± 0.117 | Yes | 20 forward passes |
| Energy OOD | 0.576 ± 0.148 | Yes | Failed |
| sFRC GT | 0.432 ± 0.076 | No | Failed |

Hallucination ground truth is defined via Bhadra et al.'s null-space decomposition: the component of the reconstruction that has zero support in measured k-space.

## Findings

![PSF vs Hallucination](figures/figures_notebook_3/fig_004.png)

*PSF aliasing pattern predicts hallucination location with r = 0.95. Control experiment confirms this is mask-specific (drop to r = 0.66 with different mask).*

U-Net hallucinations constitute ~86% of reconstruction error at 4x acceleration. They concentrate in high frequencies (13.6x enrichment) and correlate strongly with the sampling PSF (r=0.95), confirming hallucinations are physics-driven, not anatomy-driven. Deep ensembles are the strongest single GT-free detector, while TTA flip achieves comparable performance at lower cost. MC dropout with standard rates (p=0.05) underperforms, and both energy-based OOD and spectral FRC methods fail entirely. Rankings are stable across 4x and 8x acceleration.

## Methods

**Null-space decomposition** (Bhadra et al., 2021) separates reconstruction error into data-consistent and hallucinated components. The hallucination map serves as ground truth for all detector evaluations.

**Physics-informed detectors** exploit the forward model: multi-mask disagreement probes sensitivity to sampling pattern changes, IFFT gradient measures deviation from the adjoint solution, and data consistency quantifies k-space residuals.

**Uncertainty detectors** use stochastic inference: MC dropout, deep ensembles, and TTA horizontal flip.

**Negative results** are documented: energy-based OOD detection (classification transfer fails for pixel-level tasks) and reference-free sFRC (frequency correlation saturates, inverted direction).

![Detector Ranking](figures/figures_notebook_5/fig_001.png)

*Full detector ranking across 19 methods and 6 families.*

## Dataset

[fastMRI](https://fastmri.med.nyu.edu/) single-coil knee: 973 training volumes, 199 validation volumes (7,135 slices). Random Cartesian masks at 4x and 8x acceleration with 8% center fraction. Must be downloaded separately after registration.

## Installation
```bash
git clone https://github.com/GuillaumeEsclozas/mri-hallucination-physics.git
cd mri-hallucination-physics
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch 2.0+, and a CUDA GPU.

## Usage

Five notebooks reproduce all results end-to-end on Google Colab (A100/L4 GPU):

| Notebook | Content |
|---|---|
| `01_data_exploration.ipynb` | Pipeline verification, adjoint consistency, null-space validation |
| `02_reconstruction_baselines.ipynb` | U-Net training (4x/8x), PSNR/SSIM evaluation |
| `03_hallucination_characterization.ipynb` | Null-space decomposition, PSF correlation, frequency analysis |
| `04_physics_informed_detection.ipynb` | 8 physics/learned detectors, patch-level AUROC |
| `05_uncertainty_benchmark.ipynb` | 19 detectors, 6 families, cross-acceleration, combinations |

## References

- Bhadra et al. (2021). *On hallucinations in tomographic image reconstruction.* IEEE TMI, 40(11).
- Peck, Bugday & Saeys (2026). *Triggering hallucinations in model-based MRI reconstruction via adversarial perturbations.* arXiv:2602.18536.
- Theunissen, Mortier, Saeys & Waegeman (2025). *Evaluation of out-of-distribution detection methods.* Briefings in Bioinformatics, 26(3).
- Gottschling, Antun, Hansen & Adcock (2025). *The troublesome kernel.* SIAM Review, 67:73-104.
- Kc, Zeng, Soni & Badano (2024). *sFRC for assessing hallucinations in medical image restoration.* FDA/CDRH.
- Lustig, Donoho & Pauly (2007). *Sparse MRI.* MRM, 58(6).
- Zbontar et al. (2018). *fastMRI: An open dataset and benchmarks for accelerated MRI.* arXiv.
- Shimron, Tamir, Wang & Lustig (2022). *Implicit data crimes.* PNAS, 119(13).
- Ronneberger, Fischer & Brox (2015). *U-Net.* MICCAI.
- Lakshminarayanan et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles.* NeurIPS.

## License

MIT

## Acknowledgments

Built on the fastMRI dataset (NYU Langone). Hallucination framework follows Bhadra et al. and extends the detection methodology from Jonathan Peck and Lauren Theunissen's work at SaeysLab, UGent.
