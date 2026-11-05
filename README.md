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
current-state score selected by `model.uncer_metric` (`wasser` by default), a
lagged EMA of the first-step 95th percentile (decay 0.999), and a minimum of 8 steps.
Returns bootstrap at the truncation boundary; actor and imagined value losses use
the retained prefix. `model.dynamic_trunc.warmup_steps=50000` delays masking until
the trainer's logged `Step` reaches 50k (summed across environments, including
`action_repeat`, not optimizer updates). During warmup, uncertainty and threshold
EMA still update, while returns and losses use the full horizon. Set it to `0`
for immediate truncation. In `uv_run.sh`, edit `trunc_warmup_steps`.
`train/dynamic_trunc/active` logs whether truncation is currently active.
To use the full imagination horizon throughout training:

```bash
python3 train.py model.rep_loss=dreamer model.dynamic_trunc.enabled=false
```

Advantage-aware truncation is optional. The base config and older configs default
to `model.dynamic_trunc.advantage.mode=off`; `uv_run.sh` explicitly selects
`positive_only`. `signed` also tightens the threshold for negative advantages.
Supported uncertainty metrics are `wasser`, `cramer`, `sqrt_cramer`, and `sigma`;
`ucb` is rejected when this adjustment is enabled because its score can be negative.

The gate first computes full-horizon lambda returns (using the existing lambda,
normally 0.95, and bootstrap at H) and `delta_full = return_full - value` on the
same imagined trajectory and randomly selected value pair. An independent EMA
normalizes this advantage without mean subtraction. It uses the previous EMA of
the batch/time absolute-advantage quantile (default 0.9), decay 0.99 and a scale
floor of 1e-6; the first valid batch initializes the EMA. Nonfinite entries are
excluded from this quantile, and an entirely invalid batch leaves the EMA unchanged.

For normalized full advantage `a` and original uncertainty `U`, the decision score is
`U / max_factor ** tanh(a)` in `signed` mode. `positive_only` replaces
`a` with `max(a, 0)`. The default `max_factor=2.0` bounds the multiplier to
`[0.5, 2]` for `signed`, or `[0.5, 1]` for `positive_only`. This dimensionless
multiplier preserves the units of the uncertainty score, including symlog-bin
distances. `max_factor=1` returns the original uncertainty exactly, even for invalid
advantages. Otherwise nonfinite advantages produce an invalid score that crosses
the threshold, subject to the forced minimum prefix.

The threshold EMA still uses **original uncertainty**, not the adjusted score.
After deciding the mask, actor/critic learning uses the original truncated-return
procedure and the actor's existing return normalization (updated only once).
During warmup, both EMAs update, all positions are retained, and the already
computed full return is reused. `off` adds no gate return computation or model
buffers. Enabled modes save the independent scale EMA in the model state dict;
old configs/checkpoints remain strictly loadable in `off` mode. Loading an old
checkpoint into an enabled mode requires explicitly initializing the new scale
state; this does not add a complete experiment-resume facility.

```bash
python3 train.py model.distance_space=symlog model.dynamic_trunc.advantage.mode=positive_only
python3 train.py model.distance_space=symlog model.dynamic_trunc.advantage.mode=signed \
  model.dynamic_trunc.advantage.max_factor=2.0
```

The launcher exposes `adv_mode`, `adv_max_factor`, `adv_scale_quantile`,
`adv_scale_ema`, and `adv_scale_eps`, forwards them to every worker, and includes
mode/factor in the method name, for example
`VE_wasser_symlog_advpositive_only_x2.0_w50k_f64_H15`.

When enabled, `train/imag_adv_full` logs normalized full advantage and
`train/imag_trunc_score` logs adjusted uncertainty, each as a `[H,1]` curve.
Existing uncertainty curves and `train/adv` retain their meanings; the latter
still averages the training advantage over all positions, including masked ones.
Additional `train/dynamic_trunc/` scalars are:

- `adv_scale`: the lagged normalization scale actually used by the gate.
- `raw_effective_horizon`: horizon from the original U-only mask at the same threshold.
- `positive_adv_drop_fraction` / `negative_adv_drop_fraction`: removed full-advantage
  mass divided by total mass of that sign, weighted by the detached imagination
  weights. Nonfinite inputs are excluded; zero mass gives zero. During warmup,
  both masks retain all positions and both drop fractions are zero.

For ablations, compare `off`, `positive_only`, `signed`, and an unadjusted score
with a threshold scale chosen to match average effective horizon. Inspect both
evaluation returns and discarded positive-advantage mass. Historical comparison
is restricted to `VE_wasser_symlog_f64_H15`, whose runs did not have the truncation
warmup: use `model.dynamic_trunc.warmup_steps=0` for that comparison, or rerun all
groups with the same 50k warmup. These settings do not launch experiments by themselves.

`model.distance_space=raw` (default) uses the original return bins for `wasser`,
`cramer`, and `sqrt_cramer`. Set it to `symlog` to integrate CDF differences with
widths `symlog(bins).diff()` instead. The default critic's 255 bins are uniformly
spaced in symlog coordinates, reducing the weighting of tail discrepancies.
Probabilities stay unchanged; tail probability differences still contribute.
This option does not affect `ucb` or `sigma`. In `uv_run.sh`, set
`distance_space=symlog`; use a distinct `METHOD_TEMPLATE` for each experiment.

```bash
python3 train.py model.uncer_metric=wasser model.distance_space=symlog
python3 train.py model.uncer_metric=cramer model.distance_space=symlog
python3 train.py model.uncer_metric=sqrt_cramer model.distance_space=symlog
```

`wasser` uses mean pairwise Wasserstein-1 distance in the selected space.
`cramer` uses the squared Cramer score, while `sqrt_cramer` uses the Cramer
distance. Both compare full-distribution CDFs using the same selected bin
coordinates as W1 (`x` below is either raw return or symlog return):

\[
C_{ij}^{2}=\sum_{b=0}^{D-2}(F_i(b)-F_j(b))^2(x_{b+1}-x_b),\qquad
u_{\mathrm{cramer}}=\frac{1}{\binom K2}\sum_{i<j}C_{ij}^{2},\qquad
u_{\mathrm{sqrt\_cramer}}=\frac{1}{\binom K2}\sum_{i<j}\sqrt{C_{ij}^{2}}.
\]

For `sqrt_cramer`, each pair is square-rooted before averaging. Both scores
retain individual trajectories and imagination depths for truncation, are
detached float32, and use the existing threshold EMA and prefix masks.
No epsilon or extra normalization is added; identical distributions give zero.

```bash
python3 train.py model.rep_loss=dreamer model.uncer_metric=cramer
python3 train.py model.rep_loss=dreamer model.uncer_metric=sqrt_cramer
```

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
python3 train.py model.rep_loss=dreamer model.uncer_metric=ucb model.imag_horizon=30
```

For a single-head baseline, set `model.dynamic_trunc.enabled=false` and
`model.value_ensemble_size=1`. This uses the single head directly for bootstrap
and advantage. W1, sigma, cramer, and sqrt_cramer are zero; UCB equals the
single prediction (its standard deviation is defined as zero).
Dynamic truncation requires at least two heads.

`model.uncer_metric=sigma` uses only the sample standard deviation of the heads'
scalar values, in raw return units. It uses the same truncation mechanism;
`model.beta` does not affect sigma.

Only the selected score is computed and logged: `train/imag_wasser`,
`train/imag_ucb`, `train/imag_sigma`, `train/imag_cramer`, or
`train/imag_sqrt_cramer` in `metrics.jsonl` is an `[H, 1]` array averaged over
imagination starts, excluding the final bootstrap state. TensorBoard records
each depth under the selected metric as `step_00`, `step_01`, etc.
Cramer scores are independent of `model.beta`, which is used only by UCB.
The scalar `train/dynamic_trunc/effective_horizon` reports the mean retained length.

In `run.sh`, `uncer_metric` and `imag_hori` are forwarded to each worker as
`model.uncer_metric` and `model.imag_horizon`. `beta` is read from the model config
(`configs/model/_base_.yaml`), rather than overridden by the launch script. The existing
`enable_dny_trunc` and `val_ens_num` settings continue to control truncation and
head count.

Ensemble uncertainty calculations and metric selection live in `truncation.py`.
The entry point is `cal_ens_uncertainty(imag_value_dist, imag_value, metric, *, beta=0.1)`.
Here `imag_value` contains all heads with shape `[K, B*T, H+1, 1]`, rather than
the two-head bootstrap mean. The result is detached float32 `[B*T, H, 1]`.
Use `model.uncer_metric` for Hydra overrides and `uncer_metric` in the launch
scripts; the metric values and existing log names are unchanged.

`PROCS_PER_GPU` in `run.sh` defaults to `1` and limits concurrent training jobs
on each GPU. Set it to `2` to run up to two jobs per GPU. Each worker processes
its assigned queue sequentially; the tmux session is deleted after all workers exit.

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).

For offline physics-state decoding from frozen Dreamer features, see
[`docs/physics_probe.md`](docs/physics_probe.md). The probe provides separate state-restoration,
data-collection, decoder-training, and posterior/imagination evaluation commands.


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
