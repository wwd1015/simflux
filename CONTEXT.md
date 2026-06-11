# SimFlux — Credit Portfolio Domain

The shared language for SimFlux's credit-loss simulation. This file is a glossary,
not a spec: it pins what each term *means* so the docs, the Rust backend, and the
NumPy fallback all use words the same way. Implementation lives in the code.

## Language

**Two-Factor Model**:
A credit-loss model in which each obligor's latent asset value is the sum of exactly
two random pieces — a **systematic factor** and an **idiosyncratic factor**. The "two"
are systematic + idiosyncratic, per obligor.
_Avoid_: framing it as "global + sector + idiosyncratic" — there is no separate global
factor (see Flagged ambiguities).

**Systematic factor**:
The single common shock an obligor loads on: its **sector factor**. An obligor's
sensitivity to it is the sector loading (`sqrt(intra-sector correlation)`).
_Avoid_: Global factor, market factor.

**Idiosyncratic factor**:
The obligor-specific shock, drawn independently for every obligor and every period.
Its loading is `sqrt(1 − intra-sector correlation)`.
_Avoid_: Specific risk, noise.

**Sector factor**:
One systematic shock per sector. Sectors are coupled to one another through the
**sector correlation matrix** — that matrix is the *sole* source of cross-sector
dependence.

**Sector correlation matrix**:
The full N×N matrix that correlates the sector factors. Required positive definite.
The single source of truth for cross-sector coupling at the Rust seam.
_Avoid_: inter-sector correlation (a single scalar) — that only ever existed as a
convenience for *building* a uniform matrix, never as a model input.

**Intra-sector correlation**:
The share of an obligor's latent variance carried by its sector factor. Per sector,
in [0, 1]. Its square root is the obligor's sector loading.

**Conditional (forward) PD**:
The per-period default probability, `(cum[k] − cum[k−1]) / (1 − cum[k−1])` (or
`1 − (1 − pd)^(1/n_periods)` for a flat PD). It is the `factor_persistence = 0`
limit of the frailty barrier.

**Default timing model**:
How multi-period default *timing* is generated. Selected via the `default_timing`
argument as a timing object (`Copula()`, `Frailty(persistence=...)`) or its string
sugar (`"copula"`, `"frailty"` — resolved to default-configured objects). The model
is a recipe: given the book's cumulative-PD matrix and loadings it produces the
**timing plan** the backends consume. Two options, both reproducing the marginal
cumulative PD exactly, differing only in
the cross-period dependence of the systematic factor:
- **Copula** (default): one latent draw per obligor for the whole horizon, compared
  against the *cumulative* PD staircase; default at first crossing (Li 2000). All
  uncertainty resolves at t=0; loss distribution is grid-invariant.
- **Frailty**: a persistent AR(1) systematic factor with fresh idiosyncratic shocks
  each period, and a per-period *calibrated barrier* preserving the marginal PD for
  any **factor persistence** (Duffie et al. 2009). `factor_persistence=0` ⇒
  independent periods; `=1` ⇒ frozen factor.

**Timing plan**:
The frozen artifact a default timing model derives for a given book: a
`(n_periods, n_assets)` threshold matrix, the per-period AR(1) coefficient
(`factor_phi`, exactly 0 for copula), and the **kernel** name. The *sole* timing
input the simulation backends consume — backends never derive thresholds
themselves. For copula the thresholds are the cumulative-PD staircase quantiles;
for frailty they are the calibrated barriers.
_Avoid_: "barriers" as the cross-seam term — calibrated barriers are the frailty
implementation detail; the plan's thresholds cover both modes uniformly.

**Kernel**:
The simulation dynamic a backend runs against the timing plan's thresholds:
`"copula"` (one frozen latent per obligor, first crossing, no fresh idiosyncratic
shocks after t=0) or `"frailty"` (AR(1) systematic factor with fresh idiosyncratic
shocks each period, first passage). A *closed two-member set*: a new kernel
requires an inner loop in both backends plus cross-validation. Distinct from the
default timing model, which is the user-facing recipe that derives the plan — two
models may share a kernel (a custom calibration emitting `"frailty"` thresholds
needs no backend change).

**Factor persistence**:
Annual autocorrelation of the systematic credit-cycle factor (frailty mode), in
`[0, 1]`; per-period coefficient is `factor_persistence^period_length`. Calibrate to
data (AR(1) on a probit-transformed default-rate series); the default 0.5 is
illustrative.
_Avoid_: calling either model "the multi-period model" — both are multi-period.

**LGD (loss given default)**:
The fraction of exposure lost when an obligor defaults. Drawn from a Beta distribution
whose mean/variance are the obligor's `lgd_mean`/`lgd_std²`, coupled to the systematic
factor via the **systematic LGD correlation**.

**Systematic LGD correlation**:
How strongly LGD moves with the cycle. Sign convention: a *positive* value encodes
**wrong-way risk** — because defaults fire in the low tail of the sector factor, LGD
loads on the *negative* of that factor, so a downturn raises both default counts and
loss severity together. Scale: this is a *direct loading* — it equals the correlation
between the LGD driver and the systematic factor. The **intra-sector correlation** is a
*squared* loading (the default driver's correlation with the factor is its square root),
so giving LGD the same cycle-sensitivity as the default driver means `ρ_lgd = √ρ_intra`,
not `ρ_lgd = ρ_intra`.
_Avoid_: "LGD correlation" unqualified — be explicit it's the LGD↔systematic-factor
coupling, not LGD↔LGD across obligors.

## Flagged ambiguities

- **"Global factor"** — appears in legacy class docstrings and in the Rust
  `systematic_factor_global` result field. Resolved: there is no distinct global
  factor; that field is a copy of the sector factor. Treat "global factor" as a
  term to *retire*, not a real model ingredient.

## Example dialogue

> **Dev:** A bank obligor and a tech obligor both defaulted this trial — do they
> share a factor?
> **Quant:** Only through the sector correlation matrix. Each loads on its own
> sector factor plus its own idiosyncratic draw. The bank and tech *sector factors*
> are correlated; the obligors aren't directly coupled, and there's no global factor
> sitting above the sectors.
> **Dev:** So if I set the whole sector matrix to identity, sectors are independent?
> **Quant:** Right — then it's a clean per-sector one-factor model with no
> cross-sector dependence at all.
