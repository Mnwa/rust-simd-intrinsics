pub fn words(n: usize, pattern: usize) -> Vec<u32> {
    let mut state = 0x6a09_e667u32;
    (0..n).map(|i| {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        match pattern {
            0 => 0, 1 => u32::MAX, 2 => if i % 2 == 0 { 0 } else { u32::MAX },
            3 => (i % 17) as u32, 4 => state | 1, _ => state,
        }
    }).collect()
}
#[allow(dead_code)]
pub fn assert_float_same(a: f32, b: f32) {
    if a.is_nan() { assert!(b.is_nan()); } else { assert_eq!(a.to_bits(), b.to_bits()); }
}
