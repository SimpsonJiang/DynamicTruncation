# Tensor Shape Hints

To make the code easier to understand, some comments include **inline tensor shape annotations** using the symbols below.

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size (e.g., number of parallel environments or replay batch size) |
| `T` | Time steps (sequence length) |
| `T_imag` | Imagination horizon (time steps in latent rollouts) |
| `A` | Action dimension |
| `E` | Encoder embedding dimension |
| `F` | RSSM feature dimension (`S*K + D`) |
| `S` | Number of stochastic groups |
| `K` | Number of discrete categories per stochastic group |
| `D` | Deterministic state dimension |
| `G` | Number of groups/blocks (e.g., BlockLinear groups) |
| `U` | Hidden units (generic intermediate dimension) |
| `M_v` | Value ensemble members: configurable N categorical, or exactly 2 mixed members |
| `Q` | Quantile locations / fixed midpoint levels in a quantile value distribution |
| `N_bins` | Fixed categorical return support locations |
| `H`, `W` | Image height and width |
| `C` | Image channels |
| `H_feat`, `W_feat`, `C_feat` | Intermediate feature map shape (spatial resolution and channels) inside CNN modules |

For value imagination, `T_imag = model.imag_horizon + 1` includes the bootstrap state:

- Categorical logits: `[B*T, T_imag, N_bins]`; quantile atoms: `[B*T, T_imag, Q]`.
- Each distribution's scalar expectation: `[B*T, T_imag, 1]`; all heads: `[M_v, B*T, T_imag, 1]`.
- Mixed mean averages the two scalar expectations in raw return units.
- Scalar loss targets: `[B*T, T_imag, 1]`; per-state CE/QH loss: `[B*T, T_imag]`.
- Uncertainty: detached float32 `[B*T, T_imag-1, 1]`, excluding the bootstrap state.
- Replay features: `[B, T, F]`; optionally detached only for quantile repval.

See [value ensemble modes](value_ensemble.md) for loss and gradient contracts.
