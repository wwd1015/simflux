use statrs::distribution::{Normal, ContinuousCDF, Continuous};

pub fn normal_cdf(x: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.cdf(x)
}

pub fn normal_pdf(x: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.pdf(x)
}

pub fn inverse_normal_cdf(p: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.inverse_cdf(p)
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

    let variance = losses.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);

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
