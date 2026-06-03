# Multi-period default timing: copula (default) and frailty

Multi-period default timing is offered as two selectable models, because the
cross-period dependence of the systematic factor is a modeling choice, not a
fact. Both reproduce the marginal cumulative PD term structure exactly and
coincide at `n_periods = 1`.

- **`"copula"`** (default) — one-factor Gaussian copula of default *times* (Li,
  2000): one frozen latent per obligor vs the cumulative-PD staircase. All
  uncertainty resolves at t=0; the loss distribution is grid-invariant.
- **`"frailty"`** — dynamic frailty (Duffie, Eckner, Horel & Saita, 2009): a
  persistent AR(1) systematic factor with fresh idiosyncratic shocks each period,
  and a per-period **calibrated barrier** that preserves the marginal PD for any
  persistence `factor_persistence`.

## Why a calibrated barrier (and not a raw AR(1))

A naive AR(1) on the independence-derived *forward* thresholds biases the
cumulative PD downward (EL drifted ~14% at ρ=0.4 in testing), because the
forward-PD derivation assumes period independence. The barrier is calibrated by a
1-D Markov forward recursion so that `P(default by k) = cum_k` holds under the
dependence — exactly the property a credit model must keep. This is verified
deterministically (`frailty.survival_curve` inverts the calibration) and by
Monte-Carlo marginal checks.

## Why "hazard" is not a named mode

The earlier independent-per-period ("hazard") model is exactly the
`factor_persistence = 0` limit of frailty, so it remains reachable without being a
separate mode. Its standalone weakness — the tail *collapses* as the horizon is
sliced more finely — is why it is not a default.

## Known caveat

Frailty is **monitoring-frequency-dependent**: fresh idiosyncratic shocks each
period make `n_periods` a modeling choice (how often default is observed), and the
tail rises with finer grids at fixed `factor_persistence`. The copula model is
grid-invariant. Pick `n_periods` to match the real observation frequency.

## Consequences

- `default_timing="copula"` is the default; multi-period (`n_periods > 1`) tail
  numbers differ from the prior independent-period behavior.
- `factor_persistence` must be calibrated to data for production use; the default
  `0.5` is illustrative.
- A future macro-conditional factor (CCAR-style) is still out of scope per
  `docs/adr/0001`.
