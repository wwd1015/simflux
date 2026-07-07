use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use std::sync::OnceLock;

/// Shared standard normal — `normal_cdf` sits on the per-default hot path, so
/// the distribution (and its parameter validation) is constructed once.
fn std_normal() -> &'static Normal {
    static STD_NORMAL: OnceLock<Normal> = OnceLock::new();
    STD_NORMAL.get_or_init(|| Normal::new(0.0, 1.0).unwrap())
}

pub fn normal_cdf(x: f64) -> f64 {
    std_normal().cdf(x)
}

pub fn normal_pdf(x: f64) -> f64 {
    std_normal().pdf(x)
}

pub fn inverse_normal_cdf(p: f64) -> f64 {
    std_normal().inverse_cdf(p)
}

pub fn calculate_default_threshold(pd: f64) -> f64 {
    if pd <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if pd >= 1.0 {
        return f64::INFINITY;
    }
    inverse_normal_cdf(pd)
}

pub fn calculate_default_probability(asset_value: f64, threshold: f64) -> f64 {
    normal_cdf(threshold - asset_value)
}

/// Inverse of the regularized incomplete beta function `I_x(a, b)` — the Beta
/// distribution quantile.
///
/// statrs 0.17's `Beta` does not specialize `ContinuousCDF::inverse_cdf`, so it
/// inherits the trait's generic 16-step bisection: ~18 full CDF evaluations per
/// call and only ~1e-4 absolute accuracy. LGD sampling calls this once per
/// default event, making it the portfolio hot path's most expensive operation.
/// Newton iteration on `beta_reg` with a bracketing bisection safeguard
/// converges to ~1e-14 in a handful of evaluations: faster *and* accurate to
/// machine precision (matching scipy's `beta.ppf`, which the NumPy backend
/// uses — closer cross-backend semantic parity).
///
/// `ln_beta_ab` is `ln B(a, b)`, precomputable per distribution so repeated
/// quantile draws don't pay the two `ln_gamma` calls each time.
pub fn beta_inverse_cdf_prepared(a: f64, b: f64, ln_beta_ab: f64, p: f64) -> f64 {
    use statrs::function::beta::beta_reg;

    debug_assert!(a > 0.0 && b > 0.0);
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    // The mean is a robust seed for the unimodal-or-monotone shapes a valid
    // (mean, variance) parameterization produces; the bracket does the rest.
    let mut x = a / (a + b);
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);

    for _ in 0..100 {
        let f = beta_reg(a, b, x) - p;
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }

        // Newton step x - f/pdf(x), in log space to survive extreme pdf values;
        // any step that leaves the (lo, hi) bracket falls back to bisection.
        let ln_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta_ab;
        let mut next = x - f * (-ln_pdf).exp();
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }

        if (next - x).abs() <= 1e-15 + 1e-13 * next {
            return next;
        }
        x = next;
    }
    x
}

/// Convenience form of [`beta_inverse_cdf_prepared`] for one-off calls.
pub fn beta_inverse_cdf(a: f64, b: f64, p: f64) -> f64 {
    use statrs::function::beta::ln_beta;
    beta_inverse_cdf_prepared(a, b, ln_beta(a, b), p)
}

pub fn beta_mean_var_to_params(mean: f64, variance: f64) -> Result<(f64, f64), String> {
    if mean <= 0.0 || mean >= 1.0 {
        return Err("Beta distribution mean must be between 0 and 1".to_string());
    }

    if variance <= 0.0 {
        return Err("Beta distribution variance must be positive".to_string());
    }

    let max_variance = mean * (1.0 - mean);
    if variance >= max_variance {
        return Err("Beta distribution variance is too large for given mean".to_string());
    }

    let alpha = mean * (mean * (1.0 - mean) / variance - 1.0);
    let beta = (1.0 - mean) * (mean * (1.0 - mean) / variance - 1.0);

    if alpha <= 0.0 || beta <= 0.0 {
        return Err("Invalid beta distribution parameters".to_string());
    }

    Ok((alpha, beta))
}

pub fn calculate_portfolio_loss_statistics(losses: &[f64]) -> PortfolioStatistics {
    let n = losses.len() as f64;
    let mean = losses.iter().sum::<f64>() / n;

    let variance = losses.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

    let std_dev = variance.sqrt();

    let mut sorted_losses = losses.to_vec();
    sorted_losses.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let var_95 = percentile_interp(&sorted_losses, 0.95);
    let var_99 = percentile_interp(&sorted_losses, 0.99);
    let var_999 = percentile_interp(&sorted_losses, 0.999);

    PortfolioStatistics {
        mean,
        std_dev,
        var_95,
        var_99,
        var_999,
        expected_shortfall_95: calculate_expected_shortfall(&sorted_losses, 0.95),
        expected_shortfall_99: calculate_expected_shortfall(&sorted_losses, 0.99),
        max_loss: sorted_losses[sorted_losses.len() - 1],
    }
}

/// Compute the p-th percentile using linear interpolation between adjacent
/// order statistics (matching NumPy's default "linear" method).
fn percentile_interp(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    // Virtual index (0-based) corresponding to quantile p
    let idx = p * (n - 1) as f64;
    let lo = (idx.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

fn calculate_expected_shortfall(sorted_losses: &[f64], confidence_level: f64) -> f64 {
    let n = sorted_losses.len();
    let cutoff_index = ((confidence_level * n as f64).ceil() as usize).min(n - 1);

    if cutoff_index >= n - 1 {
        return sorted_losses[n - 1];
    }

    let tail_losses = &sorted_losses[cutoff_index..];
    tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
}

#[derive(Debug, Clone)]
pub struct PortfolioStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub var_95: f64,
    pub var_99: f64,
    pub var_999: f64,
    pub expected_shortfall_95: f64,
    pub expected_shortfall_99: f64,
    pub max_loss: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_functions() {
        // Test standard normal properties
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!(normal_pdf(0.0) > 0.39 && normal_pdf(0.0) < 0.40);
        assert!((inverse_normal_cdf(0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_threshold() {
        let threshold_50 = calculate_default_threshold(0.5);
        assert!((threshold_50 - 0.0).abs() < 1e-10);

        let threshold_10 = calculate_default_threshold(0.1);
        assert!(threshold_10 < 0.0); // Should be negative for low PD

        let threshold_90 = calculate_default_threshold(0.9);
        assert!(threshold_90 > 0.0); // Should be positive for high PD
    }

    #[test]
    fn test_beta_inverse_cdf_matches_scipy() {
        // Reference values from scipy.stats.beta.ppf (scipy 1.16).
        let cases = [
            (2.0, 5.0, 0.3, 0.18180347131894917),
            (2.0, 5.0, 0.999, 0.818613866919134),
            (0.5, 0.5, 0.25, 0.14644660940672624),
            (8.0, 2.0, 0.01, 0.45596630775197505),
            (3.36, 4.84, 0.5, 0.402087012677324),
            (1.2, 9.7, 1e-9, 3.4989942970729985e-9),
            (5.0, 1.5, 0.97, 0.9768823593917346),
        ];
        for (a, b, p, expected) in cases {
            let got = beta_inverse_cdf(a, b, p);
            assert!(
                (got - expected).abs() <= 1e-10 * expected.max(1e-8),
                "ppf({p}; {a}, {b}) = {got}, expected {expected}"
            );
        }
        // Round-trip against statrs's own CDF across shapes and probabilities.
        use statrs::distribution::{Beta, ContinuousCDF};
        for (a, b) in [(2.0, 5.0), (0.7, 0.9), (12.0, 3.0), (1.0, 1.0)] {
            let dist = Beta::new(a, b).unwrap();
            for i in 1..100 {
                let p = i as f64 / 100.0;
                let x = beta_inverse_cdf(a, b, p);
                assert!(
                    (dist.cdf(x) - p).abs() < 1e-10,
                    "round-trip failed for a={a} b={b} p={p}"
                );
            }
        }
        // Exact edges.
        assert_eq!(beta_inverse_cdf(2.0, 5.0, 0.0), 0.0);
        assert_eq!(beta_inverse_cdf(2.0, 5.0, 1.0), 1.0);
    }

    #[test]
    fn test_beta_params() {
        let (alpha, beta) = beta_mean_var_to_params(0.4, 0.05).unwrap();
        assert!(alpha > 0.0);
        assert!(beta > 0.0);

        // Test error cases
        assert!(beta_mean_var_to_params(0.0, 0.05).is_err());
        assert!(beta_mean_var_to_params(1.0, 0.05).is_err());
        assert!(beta_mean_var_to_params(0.5, 0.3).is_err()); // Variance too large
    }

    #[test]
    fn test_percentile_interp() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_interp(&data, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile_interp(&data, 0.5) - 3.0).abs() < 1e-12);
        assert!((percentile_interp(&data, 1.0) - 5.0).abs() < 1e-12);
        assert!((percentile_interp(&data, 0.25) - 2.0).abs() < 1e-12);
    }
}
