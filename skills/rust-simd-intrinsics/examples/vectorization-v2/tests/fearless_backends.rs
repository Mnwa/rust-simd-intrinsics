#![cfg(feature = "fearless")]
mod common;
use fearless_simd::Level;
use simd_vectorization_recipes::{fearless, scalar};

fn supported_levels() -> Vec<Level> {
    let detected = Level::new();
    #[allow(unused_mut)]
    let mut levels = vec![detected];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(token) = detected.as_sse2() { levels.push(Level::Sse2(token)); }
        if let Some(token) = detected.as_sse4_2() { levels.push(Level::Sse4_2(token)); }
        if let Some(token) = detected.as_avx2() { levels.push(Level::Avx2(token)); }
        if let Some(token) = detected.as_avx512() { levels.push(Level::Avx512(token)); }
    }
    // Tokens are only obtained by the library's checked conversions.
    levels
}
#[test]
fn every_available_fearless_level_and_accumulator_count() {
    for level in supported_levels() {
        let width = fearless::native_width(level);
        eprintln!("Fearless {level:?}: native u32 lanes={width}");
        for pattern in 0..6 {
            let data = common::words(257+32, pattern);
            for offset in 0..32 {
                for n in 0..=257 {
                    let x = &data[offset..offset+n];
                    let want = scalar::sum_wrapping(x);
                    assert_eq!(fearless::sum4_at(level, x), want);
                    assert_eq!(fearless::sum_at::<1>(level, x), want);
                    assert_eq!(fearless::sum_at::<2>(level, x), want);
                    assert_eq!(fearless::sum_at::<4>(level, x), want);
                    assert_eq!(fearless::product_at(level, x), scalar::product_wrapping(x));
                    let mut doubled = x.to_vec();
                    fearless::double_at(level, &mut doubled);
                    assert_eq!(doubled, x.iter().map(|&v| v.wrapping_mul(2)).collect::<Vec<_>>());
                }
            }
        }
        // This cannot silently run only a tail, whatever the native width is.
        for n in [4*width, 4*width+1, 4097, 65537] {
            let x = common::words(n, 5);
            assert_eq!(fearless::sum_at::<4>(level, &x), scalar::sum_wrapping(&x));
        }
    }
}
#[test]
fn independent_outputs_preserve_arithmetic_graph() {
    let x = [0.0, -0.0, 2.0, -4.0, f32::MIN_POSITIVE, f32::from_bits(1)];
    let rows = [[1.0, -1.0, 0.5, -0.5]; 6];
    let want = scalar::batched_dot4(&x, &rows);
    let got = fearless::batched_dot4(&x, &rows);
    for lane in 0..4 { common::assert_float_same(want[lane], got[lane]); }
}

#[test]
fn v1_float_contracts_at_every_supported_level() {
    use fearless_simd::{dispatch, f32x4, prelude::*};
    #[inline(always)]
    fn probe<S: Simd>(simd: S) -> [u32; 4] {
        let mixed = f32x4::from_slice(simd, &[f32::NAN, -3.0, 2.0, f32::NAN]);
        assert_eq!(mixed.reduce_min_precise(), -3.0);
        assert_eq!(mixed.reduce_max_precise(), 2.0);
        let nan = f32x4::splat(simd, f32::NAN);
        assert!(nan.reduce_min_precise().is_nan());
        assert!(nan.reduce_max_precise().is_nan());
        // The exact residual is lost by a separately rounded multiply.
        let a = f32x4::splat(simd, 1.0 + f32::EPSILON);
        let b = f32x4::splat(simd, 1.0 - f32::EPSILON);
        let one = f32x4::splat(simd, 1.0);
        let residual = -f32::EPSILON * f32::EPSILON;
        assert_eq!(a.mul_add_precise(b, -one).to_array(), [residual; 4]);
        assert_eq!(a.mul_sub_precise(b, one).to_array(), [residual; 4]);
        let cancellation = f32x4::from_slice(simd, &[1e20, 1.0, -1e20, 1.0]);
        let product = f32x4::from_slice(simd, &[1e20, 1e-20, 1e20, 1e-20]);
        let zero = f32x4::from_slice(simd, &[-0.0, 0.0, -0.0, 0.0]);
        let tiny = f32x4::splat(simd, f32::from_bits(1));
        [cancellation.reduce_sum().to_bits(), product.reduce_product().to_bits(),
            zero.reduce_sum().to_bits(), tiny.reduce_sum().to_bits()]
    }
    let expected = dispatch!(Level::baseline(), simd => probe(simd));
    for level in supported_levels() {
        assert_eq!(dispatch!(level, simd => probe(simd)), expected);
    }
}

#[test]
fn score_paths_at_every_supported_level() {
    use simd_vectorization_recipes::math;
    let table = math::log_table();
    for level in supported_levels() {
        for n in 0..=257 {
            for pattern in 0..4 {
                let c: Vec<_> = (0..n).map(|i| match pattern {
                    0 => (i % 256) as u32,
                    1 => 256 + (i as u32) * 65537,
                    2 => [0, 1, 255, 256, u32::MAX][i % 5],
                    _ => 0,
                }).collect();
                let d = vec![8u8; n];
                let got = fearless::histogram_score_at(level, &c, &d, 200.0, &table);
                let want = math::histogram_score_reference(&c, &d, 200.0);
                let scale = 1.0 + c.iter().map(|&x| f64::from(x)).sum::<f64>();
                assert!((got-want).abs() <= 1e-8*scale);
            }
        }
    }
}
