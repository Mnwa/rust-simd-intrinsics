#![cfg_attr(feature = "portable", feature(portable_simd))]
//! Tiny executable API probes: dependencies must match their pinned versions.
#[cfg(feature = "fearless")]
#[test]
fn fearless_reduction_and_mask_api() {
    use fearless_simd::{dispatch, prelude::*, u32x4, Level};
    use simd_vectorization_recipes::fearless as recipes;
    #[inline(always)]
    fn probe<S: Simd>(simd: S) {
        let v = u32x4::from_slice(simd, &[1,2,3,4]);
        assert_eq!(recipes::reduce_min4(v), 1);
        assert_eq!(recipes::reduce_max4(v), 4);
        assert_eq!(recipes::reduce_sum4(v), 10);
        assert_eq!(recipes::reduce_product4(v), 24);
        assert_eq!(recipes::reduce_and4(v), 0);
        assert_eq!(recipes::reduce_or4(v), 7);
        assert_eq!(recipes::reduce_xor4(v), 4);
        let f = fearless_simd::f32x4::from_slice(simd, &[1.0,2.0,3.0,4.0]);
        assert_eq!(recipes::reduce_sum_f32x4(f), 10.0);
        assert_eq!(recipes::reduce_product_f32x4(f), 24.0);
        assert!(v.simd_eq(v).all_true());
        assert!(v.simd_eq(v).any_true());
        assert_eq!(v.simd_eq(v).to_bitmask() & 15, 15);
    }
    let level = Level::new();
    dispatch!(level, simd => probe(simd));
}

#[cfg(feature = "fearless")]
#[test]
fn fearless_v1_arrays_tokens_and_integer_operations() {
    use fearless_simd::{dispatch, i32x4, prelude::*, u32x4, Level};
    #[inline(always)]
    fn probe<S: Simd>(simd: S) {
        assert_eq!(u32x4::<S>::LEN, 4);
        let mut v = u32x4::from_slice(simd, &[0, 1, 3, u32::MAX]);
        let borrowed: &[u32; 4] = v.as_array();
        assert_eq!(*borrowed, [0, 1, 3, u32::MAX]);
        v.as_mut_array()[1] = 2;
        let owned: [u32; 4] = v.to_array();
        assert_eq!(owned, [0, 2, 3, u32::MAX]);
        assert_eq!(v.reverse().to_array(), [u32::MAX, 3, 2, 0]);
        assert_eq!(v.count_ones().to_array(), owned.map(u32::count_ones));
        assert_eq!(v.count_zeros().to_array(), owned.map(u32::count_zeros));
        let one = u32x4::splat(v.token(), 1);
        assert_eq!(v.saturating_add(one).to_array(), owned.map(|x| x.saturating_add(1)));
        assert_eq!(v.saturating_sub(one).to_array(), owned.map(|x| x.saturating_sub(1)));
        let signed = [i32::MIN, -1, 0, i32::MAX];
        let signed_v = i32x4::from_slice(simd, &signed);
        assert_eq!(signed_v.abs().to_array(), signed.map(i32::wrapping_abs));
        let signed_counts: i32x4<S> = signed_v.count_ones();
        assert_eq!(signed_counts.to_array(), signed.map(|x| x.count_ones() as i32));
        let mask = v.simd_eq(u32x4::splat(simd, 0));
        assert_eq!(mask.reverse().to_bitmask(), 8);
        assert_eq!(mask.rotate_elements_right::<1>().to_bitmask(), 2);
        assert_eq!(mask.rotate_elements_left::<1>().to_bitmask(), 8);
        let native = <u32 as SimdIntElement>::Native::<S>::splat(mask.token(), 2);
        assert_eq!(native.reduce_sum(), 2 * S::u32s::LEN as u32);
        let floats = <f32 as SimdFloatElement>::Native::<S>::splat(simd, 1.0);
        assert_eq!(floats.reduce_product(), 1.0);
    }
    dispatch!(Level::new(), simd => probe(simd));
}

#[cfg(feature = "wide")]
#[test]
fn wide_concrete_f32x8_reducers() {
    let v = wide::f32x8::from([1.0,2.0,3.0,4.0,1.0,1.0,1.0,1.0]);
    assert_eq!(v.reduce_add(), 14.0);
    assert_eq!(v.reduce_mul(), 24.0);
    use simd_vectorization_recipes::wide_examples as w;
    assert_eq!(w::product_reassociated(&[]), 1.0);
    for n in 0..=257 {
        let x = vec![1.0f32; n];
        assert_eq!(w::sum_reassociated(&x), n as f32);
        assert_eq!(w::product_reassociated(&x), 1.0);
    }
}

#[cfg(feature = "portable")]
#[test]
fn portable_numeric_and_mask_reducers() {
    use std::simd::{prelude::*, Simd};
    let v = Simd::<u32,4>::from_array([1,2,3,4]);
    assert_eq!(v.reduce_sum(), 10); assert_eq!(v.reduce_product(), 24);
    assert_eq!(v.reduce_min(), 1); assert_eq!(v.reduce_max(), 4);
    assert_eq!(v.reduce_and(), 0); assert_eq!(v.reduce_or(), 7); assert_eq!(v.reduce_xor(), 4);
    assert!(v.simd_eq(v).any()); assert!(v.simd_eq(v).all());
    let all_nan = Simd::<f32,4>::splat(f32::NAN);
    assert!(all_nan.reduce_min().is_nan()); assert!(all_nan.reduce_max().is_nan());
    let mixed = Simd::<f32,4>::from_array([f32::NAN, -0.0, 0.0, 2.0]);
    assert_eq!(mixed.reduce_min(), 0.0); // Either sign permitted by this API.
    assert_eq!(mixed.reduce_max(), 2.0);
    assert_eq!(Simd::<f32,4>::splat(2.0).reduce_sum(), 8.0);
    assert_eq!(Simd::<f32,4>::splat(2.0).reduce_product(), 16.0);
}
