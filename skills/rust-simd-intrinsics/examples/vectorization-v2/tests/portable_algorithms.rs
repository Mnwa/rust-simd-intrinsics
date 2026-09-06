#![cfg(feature = "portable")]
mod common;
use simd_vectorization_recipes::{math, portable as p, scalar as s};

#[test]
fn reductions_scans_indices_and_fusion() {
    for pattern in 0..6 {
        let data = common::words(257+32, pattern);
        for offset in 0..32 {
            for n in 0..=257 {
                let x = &data[offset..offset+n];
                assert_eq!(p::sum_wrapping(x), s::sum_wrapping(x));
                assert_eq!(p::product_wrapping(x), s::product_wrapping(x));
                assert_eq!(p::prefix_sum(x), s::prefix_sum(x));
                assert_eq!(p::argmin(x), s::argmin(x));
                assert_eq!(p::argmax(x), s::argmax(x));
                assert_eq!(p::statistics_u32(x), s::statistics_u32(x));
                for threshold in [0, 7, u32::MAX] { assert_eq!(p::filter_gt(x, threshold), s::filter_gt(x, threshold)); }
            }
        }
    }
    // Same minimum at multiple lanes, later blocks and the scalar tail.
    for pos in 0..65 {
        let mut x = vec![u32::MAX; 65]; x[pos] = 0;
        assert_eq!(p::argmin(&x), Some((0,pos)));
        x[64] = 0; assert_eq!(p::argmin(&x), Some((0,pos.min(64))));
    }
}
#[test]
fn masks_classification_checksums_and_popcount() {
    let bytes: Vec<_> = (0..512).map(|i| i as u8).collect();
    for offset in 0..32 {
        for n in 0..=257 {
            let x = &bytes[offset..offset+n];
            let y: Vec<_> = x.iter().map(|&b| b^0x55).collect();
            assert_eq!(p::sum_u8_widened(x), s::sum_u8_widened(x));
            assert_eq!(p::hamming(x, &y), s::hamming(x, &y));
            assert_eq!(p::classify_hex(x), s::classify_hex(x));
            assert_eq!(p::ascii_uppercase(x), s::ascii_uppercase(x));
            assert_eq!(p::adler32(x), s::adler32(x));
            assert_eq!(p::common_prefix(x, &y), s::common_prefix(x, &y));
            assert_eq!(p::common_prefix(x, x), x.len());
            for needle in [0, 1, 127, 255] {
                assert_eq!(p::find_byte(x, needle), s::find_byte(x, needle));
                assert_eq!(p::count_byte(x, needle), s::count_byte(x, needle));
            }
        }
    }
    for n in [4097,65537] { let x=vec![255;n]; assert_eq!(p::adler32(&x), s::adler32(&x)); }
    for mismatch in 0..65 {
        let a = vec![1u8;65]; let mut b=a.clone(); b[mismatch]=2;
        assert_eq!(p::common_prefix(&a,&b), mismatch);
    }
}
#[test]
fn sorting_network_exhaustive_small_domain() {
    for a in 0..4 { for b in 0..4 { for c in 0..4 { for d in 0..4 {
        let x=[a,b,c,d]; let mut want=x; want.sort_unstable(); assert_eq!(p::sort4(x),want);
    }}}}
    assert_eq!(p::sort4([u32::MAX,0,7,0]), [0,0,7,u32::MAX]);
}
#[test]
fn fp_lane_axis_and_stencil_contract() {
    for n in 0..=257 {
        let x: Vec<_> = (0..n).map(|i| (i as f32 - 64.0)*0.25).collect();
        let rows: Vec<_> = (0..n).map(|i| [1.0,-1.0,0.5,(i%7) as f32]).collect();
        let want=s::batched_dot4(&x,&rows); let got=p::batched_dot4(&x,&rows);
        for lane in 0..4 { common::assert_float_same(want[lane],got[lane]); }
        assert_eq!(p::convolve3(&x,[0.5,-1.0,2.0]),s::convolve3(&x,[0.5,-1.0,2.0]));
    }
    let x=[f32::NAN,f32::INFINITY,f32::NEG_INFINITY,0.0,-0.0,f32::from_bits(1)];
    let a=p::convolve3(&x,[1.0,0.0,-1.0]); let b=s::convolve3(&x,[1.0,0.0,-1.0]);
    for (&a,&b) in a.iter().zip(&b) { common::assert_float_same(a,b); }
}
#[test]
fn all_small_all_large_mixed_log_paths_and_tails() {
    let table=math::log_table();
    for n in 0..=257 {
        for pattern in 0..4 {
            let c:Vec<_>=(0..n).map(|i| match pattern {
                0 => (i%256) as u32, 1 => 256+(i as u32)*65537,
                2 => [0,1,255,256,u32::MAX][i%5], _=>0,
            }).collect();
            let d=vec![8u8;n];
            let a=p::histogram_score(&c,&d,200.0,&table);
            let b=math::histogram_score_reference(&c,&d,200.0);
            let scale=1.0+c.iter().map(|&x| f64::from(x)).sum::<f64>();
            assert!((a-b).abs()<=1e-8*scale,"n={n}, pattern={pattern}");
        }
    }
}
