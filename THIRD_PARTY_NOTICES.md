# Third-party notices

## NVIDIA NemotronLabs RE-USE inference code

`scripts/gpu/torch_reference.py` and `scripts/mlx_enhance.py` reproduce the
pre- and post-processing of `inference.py` and `inference_chunk.py` from
[nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/tree/31f82f2efad1095034492f236e8dac9239ba713b)
(STFT parameter scaling, the sweep-artifact filter, length alignment, and the
Hann-weighted chunk overlap-add). `scripts/gpu/torch_reference.py` imports the
model definition and utilities from that repository at runtime; they are not
redistributed here.

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1

- License: <https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1>

`scripts/plot_architecture.py` draws a diagram of the model structure described
by the same repository.

The model weights and the sample audio are not redistributed by this
repository. `results/figures/mic_test2_official.png` and
`results/figures/mic_test2_mlx.png` are spectrograms of the repository's
sample audio (`noisy_audio/mic_test2.wav`) and of the model's outputs for it.

## Mamba selective scan

The Metal kernel in `scripts/metal_scan.py` is an independent implementation,
but it follows the recurrence of `selective_scan_ref` in
[state-spaces/mamba](https://github.com/state-spaces/mamba) (via its MLX port in
mlx-speech).

Copyright 2023 Tri Dao, Albert Gu

Licensed under the Apache License, Version 2.0. You may obtain a copy of the
License at <https://www.apache.org/licenses/LICENSE-2.0>.

## mlx-speech

The MLX runtime depends on [appautomaton/mlx-speech](https://github.com/appautomaton/mlx-speech)
(commit `503bd7efa9102a8e3937441448d82ae5371e679b`), installed as a dependency
and not redistributed here. `scripts/convert_mlx.py` calls its weight-remapping
functions in the same way as its `scripts/convert/reuse.py`.

Copyright (c) 2026 AppAutomaton swarm of agents

- License: MIT (<https://github.com/appautomaton/mlx-speech/blob/503bd7efa9102a8e3937441448d82ae5371e679b/LICENSE>)
