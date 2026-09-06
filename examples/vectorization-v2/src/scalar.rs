//! Scalar-written reference implementations. Autovectorization stays enabled.

pub fn sum_wrapping(input: &[u32]) -> u32 {
    input.iter().copied().fold(0, u32::wrapping_add)
}
pub fn product_wrapping(input: &[u32]) -> u32 {
    input.iter().copied().fold(1, u32::wrapping_mul)
}
pub fn sum_u8_widened(input: &[u8]) -> u64 {
    input.iter().fold(0u64, |a, &x| a.wrapping_add(u64::from(x)))
}
pub fn prefix_sum(input: &[u32]) -> Vec<u32> {
    let mut carry = 0u32;
    input.iter().map(|&x| { carry = carry.wrapping_add(x); carry }).collect()
}
pub fn argmin(input: &[u32]) -> Option<(u32, usize)> {
    input.iter().copied().enumerate().map(|(i, x)| (x, i)).min()
}
pub fn argmax(input: &[u32]) -> Option<(u32, usize)> {
    let mut best = None;
    for (i, &x) in input.iter().enumerate() {
        if best.map_or(true, |(value, _)| x > value) { best = Some((x, i)); }
    }
    best
}
pub fn find_byte(input: &[u8], needle: u8) -> Option<usize> {
    input.iter().position(|&x| x == needle)
}
pub fn count_byte(input: &[u8], needle: u8) -> usize {
    input.iter().filter(|&&x| x == needle).count()
}
pub fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}
pub fn filter_gt(input: &[u32], threshold: u32) -> Vec<u32> {
    input.iter().copied().filter(|&x| x > threshold).collect()
}
pub fn hamming(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0u64, |s, (&x, &y)| s.wrapping_add(u64::from((x ^ y).count_ones())))
}
pub fn classify_hex(input: &[u8]) -> Vec<u8> {
    input.iter().map(|&x| match x {
        b'0'..=b'9' => x - b'0', b'a'..=b'f' => x - b'a' + 10,
        b'A'..=b'F' => x - b'A' + 10, _ => 255,
    }).collect()
}
pub fn ascii_uppercase(input: &[u8]) -> Vec<u8> {
    input.iter().map(u8::to_ascii_uppercase).collect()
}
pub fn histogram(input: &[u8]) -> [usize; 256] {
    let mut bins = [0usize; 256];
    for &x in input { bins[usize::from(x)] += 1; }
    bins
}
/// Private updates avoid conflicts. The update phase is deliberately scalar.
pub fn histogram_private(input: &[u8]) -> [usize; 256] {
    let mut bins = [[0usize; 256]; 4];
    let mut chunks = input.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 { bins[lane][usize::from(chunk[lane])] += 1; }
    }
    for &x in chunks.remainder() { bins[0][usize::from(x)] += 1; }
    std::array::from_fn(|i| bins.iter().map(|part| part[i]).sum())
}
pub fn adler32(input: &[u8]) -> u32 {
    let (mut s1, mut s2) = (1u32, 0u32);
    for &x in input { s1 = (s1 + u32::from(x)) % 65521; s2 = (s2 + s1) % 65521; }
    (s2 << 16) | s1
}
/// Rows are input-major: each SIMD lane can hold one independent output.
pub fn batched_dot4(x: &[f32], rows: &[[f32; 4]]) -> [f32; 4] {
    assert_eq!(x.len(), rows.len());
    let mut out = [0.0f32; 4];
    for (&v, row) in x.iter().zip(rows) {
        for j in 0..4 { out[j] += v * row[j]; }
    }
    out
}
pub fn convolve3(input: &[f32], weights: [f32; 3]) -> Vec<f32> {
    input.windows(3).map(|v| (v[0]*weights[0] + v[1]*weights[1]) + v[2]*weights[2]).collect()
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Statistics {
    pub sum: u64,
    pub sum_squares: u64,
    pub min: Option<u32>,
    pub max: Option<u32>,
}
pub fn statistics_u32(input: &[u32]) -> Statistics {
    let mut s = Statistics { sum: 0, sum_squares: 0, min: None, max: None };
    for &x in input {
        let v = u64::from(x);
        s.sum = s.sum.wrapping_add(v);
        s.sum_squares = s.sum_squares.wrapping_add(v.wrapping_mul(v));
        s.min = Some(s.min.map_or(x, |a| a.min(x)));
        s.max = Some(s.max.map_or(x, |a| a.max(x)));
    }
    s
}
