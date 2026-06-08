#!/usr/bin/env python3
"""
Portfolio Risk Example
======================

This example demonstrates portfolio loss simulation using SimFlux.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import simflux as sf

def main():
    print("SimFlux Portfolio Risk Example")
    print("=" * 40)

    # Create sector correlation matrix
    sector_corr = np.array([
        [1.0, 0.2, 0.1],    # Technology correlations
        [0.2, 1.0, 0.15],   # Finance correlations
        [0.1, 0.15, 1.0]    # Healthcare correlations
    ])

    print("Creating portfolio with 3 sectors:")
    print("  Technology: 30 assets")
    print("  Finance: 25 assets")
    print("  Healthcare: 20 assets")
    print()

    # Create portfolio
    portfolio = sf.CreditPortfolio.create_sample_portfolio(
        n_assets_per_sector=[30, 25, 20],
        sectors=['Technology', 'Finance', 'Healthcare'],
        sector_correlation_matrix=sector_corr,
        intra_sector_correlations={
            'Technology': 0.4,    # High intra-sector correlation
            'Finance': 0.5,       # Higher correlation in finance
            'Healthcare': 0.3     # Lower correlation in healthcare
        }
    )

    # Get portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"Portfolio Summary:")
    print(f"  Total assets: {summary['n_assets']}")
    print(f"  Sectors: {summary['n_sectors']}")

    for sector, info in summary['sectors'].items():
        avg_pd = info.get('avg_pd', 0.03)
        avg_lgd = info.get('avg_lgd', 0.5)
        print(f"  {sector}: {info['n_assets']} assets, "
              f"PD={avg_pd:.1%}, LGD={avg_lgd:.1%}")

    print(f"\nRunning Monte Carlo simulation...")
    print(f"  Simulations: 10,000")

    # Run portfolio simulation
    results = portfolio.simulate(n_simulations=10000)

    # Display results
    if 'portfolio_statistics' in results:
        stats = results['portfolio_statistics']
        print(f"\nPortfolio Loss Statistics:")
        print(f"  Expected Loss:       ${stats['mean']:>12,.0f}")
        print(f"  Loss Std Dev:        ${stats['std_dev']:>12,.0f}")
        print(f"  95% VaR:             ${stats['var_95']:>12,.0f}")
        print(f"  99% VaR:             ${stats['var_99']:>12,.0f}")
        print(f"  99.9% VaR:           ${stats['var_999']:>12,.0f}")
        print(f"  Expected Shortfall:  ${stats['expected_shortfall_99']:>12,.0f}")
        print(f"  Maximum Loss:        ${stats['max_loss']:>12,.0f}")

    # Sector breakdown
    if 'sector_statistics' in results:
        print(f"\nSector Risk Breakdown:")
        total_var_95 = 0
        for sector, sector_stats in results['sector_statistics'].items():
            sector_var = sector_stats.get('var_95', 0)
            total_var_95 += sector_var
            print(f"  {sector:12} 95% VaR: ${sector_var:>10,.0f}")

        print(f"  {'Total':12} 95% VaR: ${total_var_95:>10,.0f}")

    # Risk insights
    if 'portfolio_statistics' in results:
        stats = results['portfolio_statistics']
        var_95 = stats['var_95']
        var_99 = stats['var_99']
        expected_loss = stats['mean']

        print(f"\nRisk Insights:")
        print(f"  Tail Risk Multiplier (99%/95% VaR): {var_99/var_95:.1f}x")
        print(f"  Unexpected Loss (95% VaR - Expected): ${var_95 - expected_loss:,.0f}")
        print(f"  Loss volatility ratio: {stats['std_dev']/expected_loss:.1f}x expected loss")

if __name__ == "__main__":
    main()