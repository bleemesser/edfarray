/// Tolerance, in samples, for treating a time as exactly on the sample grid. It absorbs
/// float noise such as `0.1 + 0.2` without moving any time that is genuinely between samples.
pub const GRID_EPS: f64 = 1e-6;

/// Index of the first sample at or after a time, where `samples` is that time already
/// multiplied by the sample rate.
///
/// Every time-to-sample conversion in the crate goes through this one rule, so a half-open
/// time range `[start, end)` selects the same samples whichever API resolves it.
pub fn first_sample_at_or_after(samples: f64) -> i64 {
    let nearest = samples.round();
    if (samples - nearest).abs() < GRID_EPS {
        nearest as i64
    } else {
        samples.ceil() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_float_noise_but_not_real_offsets() {
        assert_eq!(first_sample_at_or_after(0.30000000000000004 * 10.0), 3);
        assert_eq!(first_sample_at_or_after((0.7 - 0.1) * 10.0), 6);
        assert_eq!(first_sample_at_or_after(3.001), 4);
        assert_eq!(first_sample_at_or_after(2.999), 3);
        assert_eq!(first_sample_at_or_after(-0.5), 0);
        assert_eq!(first_sample_at_or_after(-1.5), -1);
    }
}
