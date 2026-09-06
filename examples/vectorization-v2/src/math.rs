//! Positive-u32 bounded-domain approximation; no exact-sign guarantee.
pub type LogTable = [f64; 256];
pub fn log_table() -> LogTable {
    std::array::from_fn(|n| if n == 0 { 0.0 } else { (n as f64).log2() })
}
pub fn decompose_count(n: u32) -> (f64, f64) {
    let n = n.max(1);
    let e = 31 - n.leading_zeros();
    (f64::from(e), f64::from(n) / ((1u64 << e) as f64))
}
/// n==0 returns the *contribution sentinel* 0, not mathematical log2(0).
pub fn log2_count_approx(n: u32) -> f64 {
    let (e, m) = decompose_count(n);
    let z = (m - 1.0) / (m + 1.0);
    let q = z * z;
    let mut p = 1.0 / 19.0;
    for k in (0..9).rev() { p = p * q + 1.0 / ((2 * k + 1) as f64); }
    e + (2.0 / std::f64::consts::LN_2) * z * p
}
/// Analytic truncation only: deliberately NOT called a total FP error bound.
pub fn series_truncation_bound() -> f64 {
    (2.0 / std::f64::consts::LN_2) * (1.0f64 / 3.0).powi(21) / (21.0 * (1.0 - 1.0 / 9.0))
}
pub fn histogram_score_reference(counts: &[u32], depths: &[u8], constant: f64) -> f64 {
    assert_eq!(counts.len(), depths.len());
    let mut sum = 0.0;
    for (&n, &d) in counts.iter().zip(depths) {
        if n != 0 { sum += f64::from(n) * (f64::from(d) + f64::from(n).log2()); }
    }
    constant - sum
}
pub fn histogram_score_approx(counts: &[u32], depths: &[u8], constant: f64, table: &LogTable) -> f64 {
    assert_eq!(counts.len(), depths.len());
    let mut sum = 0.0;
    for (&n, &d) in counts.iter().zip(depths) {
        let log = if n < 256 { table[n as usize] } else { log2_count_approx(n) };
        sum += f64::from(n) * (f64::from(d) + log);
    }
    constant - sum
}
