//! Criterion micro-benchmarks for the simulation kernels.
//!
//! These time the Rust kernels **without Python in the loop** — no timing-plan
//! derivation, no FFI, no NumPy marshalling — so a kernel change is measured
//! in isolation. The Python-side harnesses (`benchmarks/*.py`) measure the
//! user-visible end-to-end numbers; track both, and when they disagree the
//! difference is the Python side.
//!
//! Run with `cargo bench --no-default-features` (like `cargo test`, the
//! benches link libpython via a normally-linked binary, so the
//! `extension-module` feature must be off).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use simflux::gbm::{simulate_gbm_correlated, simulate_gbm_single};
use simflux::math_utils::{beta_inverse_cdf, inverse_normal_cdf};
use simflux::portfolio::{simulate_portfolio_losses, AssetData, PortfolioConfig};

fn bench_gbm_single(c: &mut Criterion) {
    c.bench_function("gbm_single_10k_paths_252_steps", |b| {
        b.iter(|| {
            simulate_gbm_single(
                black_box(0.05),
                black_box(0.2),
                black_box(100.0),
                10_000,
                252,
                1.0 / 252.0,
                Some(42),
            )
        })
    });
}

fn bench_gbm_correlated(c: &mut Criterion) {
    let n = 5;
    let mut corr = vec![vec![0.3; n]; n];
    for (i, row) in corr.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    c.bench_function("gbm_correlated_5_assets_2k_paths_252_steps", |b| {
        b.iter(|| {
            simulate_gbm_correlated(
                black_box(vec![0.05; n]),
                black_box(vec![0.2; n]),
                black_box(vec![100.0; n]),
                corr.clone(),
                2_000,
                252,
                1.0 / 252.0,
                Some(42),
            )
            .unwrap()
        })
    });
}

/// A 200-obligor, 2-sector book with the flat-PD staircase thresholds the
/// Python timing plan would derive (constant hazard, cumulative quantiles).
fn sample_book(
    n_assets: usize,
    n_periods: usize,
) -> (PortfolioConfig, Vec<AssetData>, Vec<Vec<f64>>) {
    let n_sectors = 2;
    let config = PortfolioConfig {
        intra_sector_correlations: vec![0.35, 0.25],
        systematic_lgd_correlations: vec![0.3, 0.3],
        sector_names: vec!["A".into(), "B".into()],
        sector_correlation_matrix: vec![vec![1.0, 0.2], vec![0.2, 1.0]],
    };
    let assets: Vec<AssetData> = (0..n_assets)
        .map(|i| AssetData {
            asset_id: i as u32,
            sector_id: (i % n_sectors) as u32,
            pd: 0.01 + 0.04 * (i as f64 / n_assets as f64),
            lgd_mean: 0.55,
            lgd_std: 0.2,
            exposure: 1_000_000.0,
            sector_name: if i % n_sectors == 0 { "A" } else { "B" }.into(),
            pd_term_structure: None,
            intra_sector_correlation: None,
            lgd_term_structure: None,
        })
        .collect();
    let thresholds: Vec<Vec<f64>> = assets
        .iter()
        .map(|a| {
            (0..n_periods)
                .map(|k| {
                    let cum = 1.0 - (1.0 - a.pd).powf((k + 1) as f64 / n_periods as f64);
                    inverse_normal_cdf(cum)
                })
                .collect()
        })
        .collect();
    (config, assets, thresholds)
}

fn bench_portfolio(c: &mut Criterion) {
    let (config, assets, thresholds) = sample_book(200, 4);

    c.bench_function("portfolio_copula_200_assets_2k_trials_4_periods", |b| {
        b.iter(|| {
            simulate_portfolio_losses(
                &config,
                &assets,
                black_box(2_000),
                4,
                1.0,
                "copula",
                0.0,
                &thresholds,
                Some(42),
                false,
                None,
                None,
            )
            .unwrap()
        })
    });

    c.bench_function("portfolio_frailty_200_assets_2k_trials_4_periods", |b| {
        b.iter(|| {
            simulate_portfolio_losses(
                &config,
                &assets,
                black_box(2_000),
                4,
                1.0,
                "frailty",
                0.5,
                &thresholds,
                Some(42),
                false,
                None,
                None,
            )
            .unwrap()
        })
    });
}

fn bench_beta_inverse_cdf(c: &mut Criterion) {
    c.bench_function("beta_inverse_cdf_lgd_shape", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for i in 1..100 {
                acc += beta_inverse_cdf(
                    black_box(2.85),
                    black_box(2.33),
                    black_box(i as f64 / 100.0),
                );
            }
            acc
        })
    });
}

criterion_group!(
    benches,
    bench_gbm_single,
    bench_gbm_correlated,
    bench_portfolio,
    bench_beta_inverse_cdf
);
criterion_main!(benches);
