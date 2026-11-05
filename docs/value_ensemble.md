# Value ensemble modes

## Configuration

```yaml
model:
  value_ensemble_type: categorical  # categorical / categorical_quantile
  value_ensemble_size: 5            # categorical mode only
  quantile:
    num_quantiles: 255
    huber_kappa: 1.0
    repval_grad: true
```

`categorical` retains the existing N-member ensemble, randomly selected pair for
imagination bootstrap/advantage, replay ensemble mean, and all existing uncertainty
metrics. Missing new fields in old saved configs select this mode. No quantile
parameters or checkpoint buffers are added to this mode.

`categorical_quantile` creates exactly two online heads in `value`: categorical at
index 0, quantile at index 1. `value_ensemble_size` in the config is ignored in this
mode; the agent's effective size and `value_ensemble_size` metric are 2. This mode
requires `uncer_metric=wasser` and a `symexp_twohot` categorical critic.

```bash
python train.py env=dmc_proprio env.task=dmc_walker_walk \
  model.value_ensemble_type=categorical_quantile \
  model.uncer_metric=wasser model.distance_space=symlog \
  model.quantile.repval_grad=true logdir=./logdir/mixed_example
```

The quantile head resolves and copies the categorical critic configuration, including
MLP depth/width, SiLU, RMSNorm and output initialization; only the output count and
distribution change. In particular, default zero output initialization gives uniform
categorical probabilities and quantile atoms all at zero. Both scalar expectations
start at zero, but their distributional W1 generally starts **nonzero**.

## Scalar values and losses

Quantile atoms are raw return locations, with equal mass and midpoint levels
`tau_i = (i + 0.5) / Q`. The training order of these levels is preserved, even if atom
locations cross. Like `TwoHot`, `Quantile.mode()` returns the expectation rather than
the statistical mode. `Quantile.mean()` is equivalent.

For both imagination and replay, mixed scalar value is

```text
V = 0.5 * (sum_j categorical_prob_j * raw_bin_j + mean_i quantile_i)
```

There is no stochastic value sample or random member selection in mixed mode. This
value is used by the existing lambda-return and advantage computations, including
the full-horizon advantage used by the optional advantage gate.

For a detached scalar target y, delta_i = y - quantile_i:

```text
Huber_kappa(delta) = 0.5 * delta^2                      if |delta| <= kappa
                   kappa * (|delta| - 0.5 * kappa)      otherwise
QH(theta, y) = mean_i |tau_i - 1[delta_i < 0]| * Huber_kappa(delta_i) / kappa
```

The quantile loss is evaluated in float32, including under autocast.
`Quantile.log_prob(y)` is **negative QH training loss, not a normalized log density**.
Both loss calls preserve the batch dimensions and remove the final atom dimension.

Each member fits the same detached scalar lambda-return plus its own detached slow
**scalar expectation**:

```text
L_cat = twohot_CE(cat, lambda_return) + twohot_CE(cat, slow_cat_mean)
L_quantile = QH(quantiles, lambda_return) + QH(quantiles, slow_quantile_mean)
L_value_or_repval = 0.5 * (L_cat + L_quantile)
```

Existing continuation weights, imagination prefix masks, loss normalization, and
global `loss_scales.value` / `loss_scales.repval` apply unchanged. Per-member metrics
`loss/value_categorical`, `loss/value_quantile`, `loss/repval_categorical` and
`loss/repval_quantile` are the weighted losses **before** ensemble averaging and the
global loss scales. They are logging entries, not additional terms in total loss.
CE and QH values have different scales; their magnitudes are not directly comparable.

This adapts heterogeneous representations from
[Diverse Projection Ensembles](https://proceedings.iclr.cc/paper_files/paper/2024/file/cfea00ddf5a5f514bd3368d639d1933d-Paper-Conference.pdf)
to the existing scalar-target Dreamer update. It does not implement distributional
Bellman mixture targets or the paper's separate exploration-bonus learners.

## Gradients and target lifecycle

Both members have independent slow heads, updated with the existing EMA frequency
and fraction. Online/frozen replicas share parameter storage; slow parameters remain
independent and excluded from the optimizer. `.to(...)` rebuilds frozen sharing.

Imagination features, lambda-return targets and slow scalar targets are detached.
Replay categorical features remain attached to the world model. With
`quantile.repval_grad=true`, quantile replay features are attached too; with `false`,
only the input to the replay quantile forward is detached. Quantile parameters still
receive their value and repval gradients. The flag does not disable categorical or
other world-model losses.

## Exact mixed W1

Categorical probabilities are recomputed from logits in float64. Quantile locations
are sorted for distance evaluation with uniform probability; training outputs are
not reordered. W1 integrates absolute CDF differences over the union of support
locations, handling duplicate locations and atoms outside categorical bounds.

`distance_space=raw` uses raw return locations. `distance_space=symlog` transforms
**both** supports before integration, leaving probabilities unchanged. This option
does not change quantile training targets or the raw-coordinate scalar expectation.

CDFs and integration use float64; summing from the closer CDF endpoint avoids
amplifying roundoff in remote, empty tails. Working rows are processed in chunks of
2048, with storage linear in C+Q rather than C*Q. The returned score is detached
float32 `[B*T, imag_horizon, 1]`; the final bootstrap state is excluded. Legacy
categorical W1/Cramer paths are retained.

Warmup, threshold EMA, advantage scaling and truncation boundary bootstrap continue
to use their existing rules. Different initial distributions and distance scales can
change the threshold trajectory; no learning-performance equivalence is assumed.

## Checkpoints and verification

Mixed checkpoints carry `_categorical_quantile_marker=1`. Loading rejects a mode
mismatch before copying the agent's parameters, even with `strict=False`. This also
prevents silently treating a two-categorical checkpoint as a mixed checkpoint when
both last layers happen to have 255 outputs. Same-mode model/optimizer roundtrips
use normal state dict loading; preserve the saved Hydra config, including Q and kappa.
This is not a new full-experiment resume mechanism.

Focused CPU tests:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_*value_ensemble.py' -v
```

Also run the existing dynamic-truncation and vector-logging suites. Tests isolate
world-model gradient paths using nonzero output weights; at default zero output
initialization, zero first-step feature gradients alone cannot validate a detach flag.
