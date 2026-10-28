## R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides the official implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a decoder-free, augmentation-free world model with redundancy-reducing regularization on latent representations. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

Installing via a virtual env like [uv](https://docs.astral.sh/uv/) is recommended. If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Select the environments you need, or use "all" to install everything.
# Available extras: dmc, atari, crafter, metaworld, memorymaze, isaaclab, all
pip install -e ".[dmc,metaworld]"
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

Monitoring results:
```bash
tensorboard --logdir ./logdir
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

The value ensemble defaults to 5 independent heads. For example:

```bash
python3 train.py model.rep_loss=dreamer model.value_ensemble_size=5
```

Each update samples two heads whose mean predicts imagination bootstrap values and the
advantage baseline. All heads train on the same return targets, with a separate slow
target per head; their losses are averaged before backpropagation.

Dynamic imagination truncation is enabled by default. Each trajectory uses its
current-state score selected by `model.VE_metric` (`wasser` by default), a
lagged EMA of the first-step 95th percentile (decay 0.999), and a minimum of 8 steps.
Returns bootstrap at the truncation boundary; actor and imagined value losses use
the retained prefix. To use the full imagination horizon:

```bash
python3 train.py model.rep_loss=dreamer model.dynamic_trunc.enabled=false
```

`wasser` uses mean pairwise Wasserstein-1 distance on the raw return bins.
`ucb` uses all heads' scalar value predictions and the sample standard deviation:

\[
\mu = \frac{1}{n}\sum_i V_i,\qquad
\sigma = \sqrt{\frac{1}{n-1}\sum_i(V_i-\mu)^2},\qquad
\mathrm{UCB} = \mu + \beta\sigma.
\]

This follows [ELVIS equations (14)-(15)](https://arxiv.org/html/2605.04709v1#S4.SS2).
`model.beta` defaults to `0.1` and must be finite and non-negative. UCB replaces
only the truncation score; it uses the same threshold EMA and prefix masks as W1.
The score is detached float32 and receives no extra normalization or clipping.
It includes the predicted value mean, so identical heads can have nonzero UCB.

```bash
python3 train.py model.rep_loss=dreamer model.VE_metric=ucb model.imag_horizon=30
```

For a single-head baseline, set `model.dynamic_trunc.enabled=false` and
`model.value_ensemble_size=1`. This uses the single head directly for bootstrap
and advantage. W1 and sigma are zero; UCB equals the
single prediction (its standard deviation is defined as zero).
Dynamic truncation requires at least two heads.

`model.VE_metric=sigma` uses only the sample standard deviation of the heads'
scalar values, in raw return units. It uses the same truncation mechanism;
`model.beta` does not affect sigma.

Only the selected score is computed and logged: `train/imag_wasser`,
`train/imag_ucb`, or `train/imag_sigma` in `metrics.jsonl` is an `[H, 1]` array
averaged over imagination starts, excluding the final bootstrap state.
TensorBoard records each depth as `train/imag_wasser/step_00`,
`train/imag_ucb/step_00`, or `train/imag_sigma/step_00`, `step_01`, etc.
The scalar `train/dynamic_trunc/effective_horizon` reports the mean retained length.

In `run.sh`, `VE_metric` and `imag_hori` are forwarded to each worker as
`model.VE_metric` and `model.imag_horizon`. `beta` is read from the model config
(`configs/model/_base_.yaml`), rather than overridden by the launch script. The existing
`enable_dny_trunc` and `val_ens_num` settings continue to control truncation and
head count.

`PROCS_PER_GPU` in `run.sh` defaults to `1` and limits concurrent training jobs
on each GPU. Set it to `2` to run up to two jobs per GPU. Each worker processes
its assigned queue sequentially; the tmux session is deleted after all workers exit.

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | State / Image | Continuous | 0.5M–1M | GPU-parallelized robotic simulation. |

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Troubleshooting

If you encounter installation or runtime issues, please refer to our [Troubleshooting Guide](docs/troubleshooting.md).

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch
