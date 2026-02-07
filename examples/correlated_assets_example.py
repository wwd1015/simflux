#!/usr/bin/env python3
"""
Correlated Assets Example
=========================

This example demonstrates multi-asset correlated simulation using SimFlux.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import simflux as sf

def main():
    print("SimFlux Correlated Assets Example")
    print("=" * 45)

    # Define correlation matrix for 3 assets
    correlation_matrix = np.array([
        [1.0, 0.5, 0.3],  # Asset 1 correlations
        [0.5, 1.0, 0.4],  # Asset 2 correlations
        [0.3, 0.4, 1.0]   # Asset 3 correlations
    ])

    print("Correlation Matrix:")
    for i, row in enumerate(correlation_matrix):
        print(f"  Asset {i+1}: {row}")

    # Create correlated GBM
    corr_gbm = sf.CorrelatedGBM(
        mu=[0.08, 0.06, 0.10],        # Different expected returns
        sigma=[0.20, 0.25, 0.30],     # Different volatilities
        S0=[100, 50, 200],            # Different starting prices
        correlation_matrix=correlation_matrix
    )

    print(f"\nAsset Parameters:")
    print(f"  Asset 1: $100, 8.0% return, 20% volatility")
    print(f"  Asset 2: $50,  6.0% return, 25% volatility")
    print(f"  Asset 3: $200, 10.0% return, 30% volatility")

    # Simulate correlated paths
    print(f"\nSimulating 1000 correlated paths for 1 year...")
    paths = corr_gbm.simulate(n_paths=1000, n_steps=252, T=1.0)
    print(f"Result shape: {paths.shape} (paths × assets × time)")

    # Analyze final returns
    print(f"\nFinal Asset Performance:")
    final_returns = []
    for i in range(3):
        initial = paths[:, i, 0].mean()
        final = paths[:, i, -1].mean()
        returns = (paths[:, i, -1] / paths[:, i, 0] - 1) * 100
        final_returns.append((paths[:, i, -1] / paths[:, i, 0] - 1))

        print(f"  Asset {i+1}: ${initial:.0f} → ${final:.0f} "
              f"({returns.mean():+.1f}% ± {returns.std():.1f}%)")

    # Check realized vs expected correlations
    realized_corr = np.corrcoef(final_returns)

    print(f"\nCorrelation Analysis:")
    print(f"  Expected vs Realized:")
    for i in range(3):
        for j in range(i+1, 3):
            expected = correlation_matrix[i, j]
            realized = realized_corr[i, j]
            diff = abs(realized - expected)
            print(f"    Asset {i+1}-{j+1}: {expected:.2f} → {realized:.2f} "
                  f"(diff: {diff:.3f})")

if __name__ == "__main__":
    main()