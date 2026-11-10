# ShellSim3D

> TODO: one-paragraph description of the project (data-driven constitutive model for reinforced concrete shells),
> authors, affiliation, and the related publication.

## Contents

| Folder | Content |
| --- | --- |
| [`sampling/`](sampling/README.md) | Generation of the training dataset: generalised strains, stresses and tangent stiffnesses of layered reinforced concrete shell sections. |
| `training/` | TODO |
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
