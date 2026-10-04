# Third-party software and research

Cyberfly's own source is licensed under Apache-2.0; see [LICENSE](LICENSE)
and [NOTICE](NOTICE). Third-party software, model assets, and datasets retain
their own licenses and attribution requirements.

## Runtime dependencies

The repository declares dependencies rather than vendoring their source or
binaries. Major components include:

- [FlyGym / NeuroMechFly](https://github.com/NeLy-EPFL/flygym),
  copyright 2023–2026 The NeuroMechFly v2 Authors, Apache-2.0.
  Cyberfly imports the fly body, hybrid locomotion controller, and supporting
  simulation utilities from this project.
- [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0,
  provides physics, rendering, and the native viewer.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), MIT,
  provides PPO and training utilities.
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), MIT,
  provides the environment interface.
- [PyTorch](https://github.com/pytorch/pytorch), primarily BSD-style with
  additional third-party license terms, provides neural networks and optimization.
- [NumPy](https://github.com/numpy/numpy) and
  [SciPy](https://github.com/scipy/scipy), primarily BSD-style with additional
  bundled-component terms, provide numerical computation.
- [ImageIO](https://github.com/imageio/imageio) and
  [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), BSD-2-Clause,
  provide video output. FFmpeg itself has separate licensing terms depending
  on the binary and its build configuration.

This list describes the main integrations, not every transitive dependency.
The authoritative license texts are in each installed distribution and its
upstream repository. `requirements-macos.lock.txt` records a tested environment;
it does not replace those license texts.

## Assets and experimental data

FlyGym supplies or downloads its model assets. They are not included in this
repository, and Cyberfly's license does not relicense them. Check the upstream
terms before redistributing assets or dependency binaries.

No experimental trajectory dataset, trained checkpoint, or simulation video is
included. Adding any of these to a release requires documenting its source,
license, experimental conditions, and any identifying metadata.

## Scientific references

[FORAGING.md](FORAGING.md#6-公开研究如何指导模型) links the research used to
guide the functional abstractions. Referencing a paper does not imply that
Cyberfly reproduces its neural model, experimental conditions, or results.
