#!/usr/bin/env python3
"""
Simple Time-Varying GBM Example
===============================

This example shows SimFlux's simple time-varying parameter functionality.
Just provide time series arrays - much simpler than the old complex API!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import simflux as sf


def main():
    print("SimFlux Simple Time-Varying Parameters Example")
    print("=" * 50)

    # Example 1: Market Crisis Scenario
    print("\n1. Market Crisis Scenario")
    print("-" * 30)

    # Define simple time series
    mu_times = [0.0, 0.25, 0.75, 1.0]
    mu_values = [0.08, -0.20, 0.15, 0.08]  # Normal -> Crisis -> Recovery -> Normal

    sigma_times = [0.0, 0.25, 0.75, 1.0]
    sigma_values = [0.18, 0.50, 0.25, 0.18]  # Low -> High -> Medium -> Low vol

    print("Time series definition:")
    print("  Times:     ", mu_times)
    print("  Mu values: ", [f"{v:+.1%}" for v in mu_values])
    print("  Sigma vals:", [f"{v:.1%}" for v in sigma_values])

    # Create GBM
    gbm = sf.TimeVaryingGBM(
        mu_times=mu_times,
        mu_values=mu_values,
        sigma_times=sigma_times,
        sigma_values=sigma_values,
        S0=100
    )

    # Check parameter evolution
    print("\nParameter evolution:")
    for t in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]:
        mu, sigma = gbm.get_parameters_at_time(t)
        print(f"  t={t:.3f}: μ={mu:+.1%}, σ={sigma:.1%}")

    # Simulate
    paths = gbm.simulate(n_paths=1000, n_steps=252, T=1.0)

    print(f"\nSimulation results:")
    print(f"  Initial price: ${paths[:, 0].mean():.2f}")
    print(f"  Crisis (t=0.5): ${paths[:, 126].mean():.2f} ± ${paths[:, 126].std():.2f}")
    print(f"  Final price: ${paths[:, -1].mean():.2f} ± ${paths[:, -1].std():.2f}")

    # Example 2: Multi-Asset with Different Patterns
    print("\n\n2. Multi-Asset Example: Tech vs Utilities")
    print("-" * 45)

    # Tech stock: high growth, high crisis impact
    tech_mu_times = [0.0, 0.3, 1.0]
    tech_mu_values = [0.15, -0.10, 0.25]  # High growth, big crisis hit
    tech_sigma_times = [0.0, 0.3, 1.0]
    tech_sigma_values = [0.30, 0.65, 0.35]  # High volatility throughout

    # Utility stock: stable, less crisis impact
    util_mu_times = [0.0, 0.4, 1.0]
    util_mu_values = [0.05, 0.02, 0.06]  # Stable returns
    util_sigma_times = [0.0, 0.4, 1.0]
    util_sigma_values = [0.12, 0.20, 0.14]  # Lower volatility

    print("Tech stock:    High growth, high volatility, crisis sensitive")
    print("Utility stock: Stable returns, low volatility, defensive")

    # Correlation: moderate correlation, higher during crisis
    correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])

    # Create multi-asset GBM
    multi_gbm = sf.TimeVaryingCorrelatedGBM(
        mu_times=[tech_mu_times, util_mu_times],
        mu_values=[tech_mu_values, util_mu_values],
        sigma_times=[tech_sigma_times, util_sigma_times],
        sigma_values=[tech_sigma_values, util_sigma_values],
        S0=[100, 100],
        correlation_matrix=correlation_matrix
    )

    # Simulate
    multi_paths = multi_gbm.simulate(n_paths=1000, n_steps=252, T=1.0)

    print(f"\nSimulation results:")
    print(f"  Tech final:    ${multi_paths[:, 0, -1].mean():.2f} ± ${multi_paths[:, 0, -1].std():.2f}")
    print(f"  Utility final: ${multi_paths[:, 1, -1].mean():.2f} ± ${multi_paths[:, 1, -1].std():.2f}")

    # Check realized correlation
    tech_returns = multi_paths[:, 0, -1] / multi_paths[:, 0, 0] - 1
    util_returns = multi_paths[:, 1, -1] / multi_paths[:, 1, 0] - 1
    realized_corr = np.corrcoef(tech_returns, util_returns)[0, 1]
    print(f"  Realized correlation: {realized_corr:.3f} (target: 0.300)")

    # Example 3: Interest Rate Cycle
    print("\n\n3. Interest Rate Cycle (5 years)")
    print("-" * 35)

    # Model 5-year interest rate cycle
    years = np.linspace(0, 5, 21)  # Quarterly points over 5 years
    base_rate = 0.06
    cycle_amplitude = 0.04

    # Sinusoidal cycle for mu
    cycle_mu = base_rate + cycle_amplitude * np.sin(2 * np.pi * years / 5)

    # Volatility also varies cyclically (inverse relationship)
    base_vol = 0.18
    vol_amplitude = 0.08
    cycle_sigma = base_vol - vol_amplitude * np.sin(2 * np.pi * years / 5)

    print(f"Interest rate cycle: {base_rate:.1%} ± {cycle_amplitude:.1%} over 5 years")
    print(f"Volatility cycle:    {base_vol:.1%} ± {vol_amplitude:.1%} (inverse)")

    # Create cyclical GBM
    cycle_gbm = sf.TimeVaryingGBM(
        mu_times=years,
        mu_values=cycle_mu,
        sigma_times=years,
        sigma_values=cycle_sigma,
        S0=100
    )

    # Show one complete cycle
    print("\nParameter evolution over cycle:")
    for year in [0, 1.25, 2.5, 3.75, 5.0]:
        mu, sigma = cycle_gbm.get_parameters_at_time(year)
        print(f"  Year {year:4.1f}: μ={mu:+.1%}, σ={sigma:.1%}")

    # Simulate 5-year period
    cycle_paths = cycle_gbm.simulate(n_paths=500, n_steps=1260, T=5.0)  # 5 years daily

    print(f"\n5-year simulation results:")
    print(f"  Year 0: ${cycle_paths[:, 0].mean():.2f}")
    print(f"  Year 5: ${cycle_paths[:, -1].mean():.2f} ± ${cycle_paths[:, -1].std():.2f}")

    print("\n" + "=" * 60)
    print("SIMPLE TIME-VARYING PARAMETERS COMPLETE!")
    print("Much easier - just provide time and value arrays!")
    print("=" * 60)


if __name__ == "__main__":
    main()