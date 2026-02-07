#!/usr/bin/env python3
"""
Basic GBM (Geometric Brownian Motion) Example
==============================================

This example demonstrates single asset simulation using SimFlux.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import simflux as sf

def main():
    print("SimFlux Basic GBM Example")
    print("=" * 40)

    # Create a GBM for a stock
    # mu = 5% annual drift, sigma = 20% annual volatility, S0 = $100 initial price
    gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
    print(f"Created GBM: μ=5%, σ=20%, S₀=$100")

    # Simulate 1000 paths for 1 year (252 trading days)
    print("Simulating 1000 paths for 1 year...")
    paths = gbm.simulate(n_paths=1000, n_steps=252, T=1.0)

    # Analyze results
    print(f"\nResults:")
    print(f"  Shape: {paths.shape} (paths × time steps)")
    print(f"  Initial price: ${paths[0, 0]:.2f}")
    print(f"  Average final price: ${paths[:, -1].mean():.2f}")
    print(f"  Expected theoretical: ${100 * np.exp(0.05):.2f}")
    print(f"  Price range: ${paths[:, -1].min():.2f} - ${paths[:, -1].max():.2f}")

    # Return statistics
    returns = (paths[:, -1] / paths[:, 0] - 1) * 100
    print(f"\nReturn Statistics:")
    print(f"  Average return: {returns.mean():.1f}%")
    print(f"  Return std dev: {returns.std():.1f}%")
    print(f"  Min return: {returns.min():.1f}%")
    print(f"  Max return: {returns.max():.1f}%")

if __name__ == "__main__":
    main()