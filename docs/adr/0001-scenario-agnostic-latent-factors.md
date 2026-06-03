# Scenario-agnostic latent systematic factors (no macro/CCAR conditioning)

Both the default model and the LGD model are driven by abstract standard-normal
systematic factors that are **not** mapped to, or calibrated against, any observable
macroeconomic variable or named scenario. We chose this deliberately: SimFlux answers an
*unconditional* question — "what loss distribution does the estimated correlation
structure imply?" — by integrating over all systematic states, rather than the
*conditional* question CCAR/DFAST and IFRS9/CECL ask — "what is the loss under the Fed's
severely adverse scenario?"

## Consequences

- A macro-conditional LGD/PD (regress on GDP, unemployment, HPI; condition on a supplied
  scenario path) is **out of scope**. Supporting it would require mapping the latent
  sector factor to macro covariates and ingesting exogenous scenario paths — a different
  framework. Macro stress testing is left to dedicated regulatory tooling.
- The systematic LGD correlation (`ρ_lgd`) is therefore a *structural* coupling parameter,
  not a regression on macro data; calibration targets are model-internal (downturn-LGD
  multiple, historical PD–LGD correlation) — see `docs/methodology.md` §2.10–2.11.
- This is hard to reverse because it is the framework's identity (a scenario-agnostic risk
  engine), which is why it is recorded here.
