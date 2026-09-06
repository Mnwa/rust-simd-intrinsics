mod common;
use simd_vectorization_recipes::{arch, math, scalar};

#[test]
fn raw_backend_matrix_reaches_vector_bodies() {
    let backends = arch::supported_backends();
    for backend in [arch::Backend::Scalar, arch::Backend::Sse2, arch::Backend::Avx2, arch::Backend::Neon] {
        if !backends.contains(&backend) { eprintln!("backend {backend:?}: unsupported"); continue; }
        for pattern in 0..6 {
            let data = common::words(257+32, pattern);
            let bytes: Vec<_> = data.iter().map(|&x| x as u8).collect();
            for offset in 0..32 {
                for n in 0..=257 {
                    let x = &data[offset..offset+n];
                    let b = &bytes[offset..offset+n];
                    assert_eq!(arch::sum_with(backend, x), scalar::sum_wrapping(x), "{backend:?}, offset={offset}, n={n}");
                    assert_eq!(arch::prefix_with(backend, x), scalar::prefix_sum(x));
                    assert_eq!(arch::sum_u8_with(backend, b), scalar::sum_u8_widened(b));
                }
            }
        }
        for n in [1023, 1024, 1025, 4097, 65537] {
            let data = common::words(n, 5);
            assert_eq!(arch::sum_with(backend, &data), scalar::sum_wrapping(&data));
            assert_eq!(arch::prefix_with(backend, &data), scalar::prefix_sum(&data));
            let bytes = vec![255; n];
            assert_eq!(arch::sum_u8_with(backend, &bytes), 255u64*(n as u64));
        }
        eprintln!("backend {backend:?}: integer sum/widen/scan passed (AVX2 scan/widen use SSE2)");
    }
}

#[test]
fn byte_search_every_lane_and_end() {
    for backend in arch::supported_backends() {
        for offset in 0..32 {
            for n in 0..=129 {
                let mut storage = vec![1u8; offset+n];
                assert_eq!(arch::find_with(backend, &storage[offset..], 99), None);
                for position in 0..n {
                    storage[offset+position] = 99;
                    assert_eq!(arch::find_with(backend, &storage[offset..], 99), Some(position));
                    storage[offset+position] = 1;
                }
                storage[offset..].fill(99);
                assert_eq!(arch::find_with(backend, &storage[offset..], 99), if n==0 {None} else {Some(0)});
            }
        }
    }
}

#[test]
fn private_histogram_handles_duplicate_indices() {
    for n in 0..=257 {
        for byte in [0, 5, 255] {
            let data = vec![byte; n];
            assert_eq!(scalar::histogram_private(&data), scalar::histogram(&data));
        }
        let data: Vec<_> = common::words(n, 5).into_iter().map(|x| x as u8).collect();
        assert_eq!(scalar::histogram_private(&data), scalar::histogram(&data));
    }
    assert_eq!(scalar::histogram_private(&[5,5])[5], 2);
}

#[test]
fn bounded_logarithm_sampling_is_not_an_error_proof() {
    assert_eq!(math::log2_count_approx(0), 0.0);
    let mut values: Vec<u32> = (1..=65536).collect();
    values.extend(common::words(100_000, 5));
    values.push(u32::MAX);
    for bit in 0..32 {
        let n = 1u32 << bit;
        values.extend([n.saturating_sub(1), n, n.saturating_add(1)]);
    }
    let mut max_error = 0.0f64;
    for n in values.into_iter().filter(|&x| x != 0) {
        let error = (math::log2_count_approx(n)-f64::from(n).log2()).abs();
        max_error = max_error.max(error);
        assert!(error <= 1e-9, "n={n}, error={error}");
    }
    eprintln!("sampled log2 max absolute error={max_error:e}; analytic truncation-only bound={:e}", math::series_truncation_bound());
}

#[test]
fn score_zero_and_mixed_tail_contract() {
    let table = math::log_table();
    for n in 0..=257 {
        let c: Vec<_> = (0..n).map(|i| [0,1,255,256,1024,u32::MAX][i%6]).collect();
        let d = vec![7u8; n];
        let exact = math::histogram_score_reference(&c, &d, 200.0);
        let approx = math::histogram_score_approx(&c, &d, 200.0, &table);
        let scale = 1.0 + c.iter().map(|&x| f64::from(x)).sum::<f64>();
        assert!((exact-approx).abs() <= 1e-8*scale);
    }
    assert_eq!(math::histogram_score_approx(&[0; 17], &[0;17], 200.0, &table), 200.0);
}

#[test]
fn scalar_oracle_known_values() {
    assert_eq!(scalar::sum_wrapping(&[u32::MAX, 2]), 1);
    assert_eq!(scalar::product_wrapping(&[]), 1);
    assert_eq!(scalar::product_wrapping(&[u32::MAX, 2]), u32::MAX-1);
    assert_eq!(scalar::argmin(&[3,1,1,2]), Some((1,1)));
    assert_eq!(scalar::argmax(&[3,3,1,2]), Some((3,0)));
    assert_eq!(scalar::adler32(b"Wikipedia"), 0x11e6_0398);
    assert_eq!(scalar::adler32(b""), 1);
}
