# ShellSim3D

Machine-Learning based Nonlinear Finite Element Analysis (ML-NLFEA) for reinforced concrete structures that can be modelled as shell elements. Includes full pipeline (sampling, training, testing) for creating differentiable structural analysis surrogate model.

> Accompanying GitHub Repo for "A Differentiable Nonlinear Finite Element Analysis for Reinforced Concrete Structures" (Draft)

## Contents

| Folder | Content |
| --- | --- |
| [`sampling/`](sampling/README.md) | Generation of the training dataset: generalised strains, stresses and tangent stiffnesses of layered reinforced concrete shell sections. |
| [`training/`](training/README.md) | Sobolev training of the neural-network surrogate (stresses and tangent stiffness), trained model of the paper and single-element tests. |
| [`deploying/`](deploying/README.md) | NN-based nonlinear FEA of RC shells vs. NLFEA: load-path table of plate case studies and the frame bridge. |

The frame bridge example can be found in the deployment folder: [`deploying/frame_bridge/`](deploying/frame_bridge) (model and usage: [deploying README, 3.2](deploying/README.md#32-frame-bridge-run_frame_bridgepy)).

<img src="deploying/frame_bridge/fb_ml_nlfea.png" alt="Frame bridge, ML-NLFEA" width="100%">

## Installation

```
conda env create -f ShellSim_vb_0_gpu.yml
conda activate ShellSim_vb_0_gpu
```

The environment uses CUDA 11.8 builds (cupy, torch). See the individual folders for how to run each part.

## Citation

How to cite: 

> Balmer, V., Näsbom, A., Thoma, K., Coros, S., Kaufmann, W. (2026). A Differentiable Nonlinear Finite Element Analysis for Reinforced 
> Concrete Structures. *Draft Version, Link tbd asap*

The layered shell FE code in this repository (shell elements, mesh generation, post-processing) is an extension of existing work of Näsbom et al. described in the following reference; no corresponding GitHub repository was published for that baseline code.

> Näsbom, A., Thoma, K., and Kaufmann, W. (2026). Numerically efficient mechanical
> modeling of reinforced concrete membrane elements with fixed, interlocked cracks. *Structural Concrete*, 1–18.
> [doi:10.1002/suco.70520](https://doi.org/10.1002/suco.70520).

## License

See [LICENSE](LICENSE).
