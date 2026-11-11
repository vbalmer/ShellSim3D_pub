# ShellSim3D

Machine-Learning based Nonlinear Finite Element Analysis (ML-NLFEA) for reinforced concrete structures that can be modelled as shell elements. Includes full pipeline (sampling, training, testing) for creating differentiable structural analysis surrogate model.

> Accompanying GitHub Repo for "A Differentiable Nonlinear Finite Element Analysis for Reinforced Concrete Structures" (Draft)

## Contents

| Folder | Content |
| --- | --- |
| [`sampling/`](sampling/README.md) | Generation of the training dataset: generalised strains, stresses and tangent stiffnesses of layered reinforced concrete shell sections. |
| [`training/`](training/README.md) | Sobolev training of the neural-network surrogate (stresses and tangent stiffness), trained model of the paper and single-element tests. |
| `deploying/` | TODO |
| `frame_bridge/` | TODO |

## Installation

```
conda env create -f ShellSim_vb_0_gpu.yml
conda activate ShellSim_vb_0_gpu
```

The environment uses CUDA 11.8 builds (cupy, torch). See the individual folders for how to run each part.

## Citation

How to cite: Paper only available as a draft so far, Link will be provided asap.

## License

See [LICENSE](LICENSE).
