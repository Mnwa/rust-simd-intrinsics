//! Small dependency-free experiment runner, not a replacement for statistical
//! benchmark infrastructure. Records every sample; no fabricated speedup claims.
use std::{hint::black_box, time::{Duration, Instant}};
use simd_vectorization_recipes::{arch, math, scalar};

fn measure<F: FnMut() -> u64>(algorithm: &str, candidate: &str, items: usize,
    samples: usize, min_time: Duration, mut f: F) {
    let mut iterations = 1u64;
    loop {
        let start = Instant::now();
        for _ in 0..iterations { black_box(f()); }
        if start.elapsed() >= min_time || iterations >= (1 << 24) { break; }
        iterations *= 2;
    }
    for sample in 0..samples {
        let start = Instant::now();
        for _ in 0..iterations { black_box(f()); }
        let ns = start.elapsed().as_secs_f64() * 1e9 / iterations as f64;
        println!("{algorithm},{candidate},{items},{iterations},{sample},{ns:.3}");
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len()>3 { return Err("usage: recipe-bench [size<=16777216] [samples>=1] [minimum_ms>=1]".into()); }
    let n = args.first().map_or(Ok(65536usize), |s| s.parse())?;
    let samples = args.get(1).map_or(Ok(7usize), |s| s.parse())?;
    let ms = args.get(2).map_or(Ok(20u64), |s| s.parse())?;
    if n>16_777_216 || samples==0 || samples>1000 || ms==0 || ms>60_000 { return Err("invalid benchmark bounds".into()); }
    let mut state=0x6a09_e667u32;
    let data:Vec<_>=(0..n).map(|_| { state^=state<<13; state^=state>>17; state^=state<<5; state }).collect();
    let bytes:Vec<_>=data.iter().map(|&x| (x%255) as u8).collect(); // Needle 255 is absent: all bytes scanned.
    #[cfg(feature="portable")]
    let other_bytes:Vec<_>=bytes.iter().enumerate().map(|(i,&x)| x ^ (i as u8).wrapping_mul(73).wrapping_add(1)).collect();
    let small:Vec<_>=data.iter().map(|&x| x%1024).collect();
    let depths=vec![8u8;n];
    let table=math::log_table();
    let duration=Duration::from_millis(ms);
    assert_eq!(arch::sum_wrapping(&data),scalar::sum_wrapping(&data));
    assert_eq!(arch::prefix_sum(&data),scalar::prefix_sum(&data));
    eprintln!("host_arch={} detected={:?}; public raw wrappers include detection/support checks",std::env::consts::ARCH,arch::detected_backend());
    eprintln!("setup/allocation of input and LUT excluded; returned Vec allocation is included; samples are observations, not significance tests");
    println!("algorithm,candidate,items,iterations,sample,ns_per_call");
    macro_rules! bench {
        ($algorithm:expr,$candidate:expr,$body:expr) => {
            measure($algorithm,$candidate,n,samples,duration,|| { $body });
        };
    }
    bench!("sum_u32","scalar_autovec",u64::from(scalar::sum_wrapping(black_box(&data))));
    bench!("sum_u32","raw_public",u64::from(arch::sum_wrapping(black_box(&data))));
    bench!("sum_u8","scalar_autovec",scalar::sum_u8_widened(black_box(&bytes)));
    bench!("sum_u8","raw_public",arch::sum_u8_widened(black_box(&bytes)));
    bench!("scan","scalar_autovec",black_box(scalar::prefix_sum(black_box(&data))).len() as u64);
    bench!("scan","raw_public",black_box(arch::prefix_sum(black_box(&data))).len() as u64);
    bench!("find_absent","scalar_autovec",scalar::find_byte(black_box(&bytes),255).unwrap_or(n) as u64);
    bench!("find_absent","raw_public",arch::find_byte(black_box(&bytes),255).unwrap_or(n) as u64);
    bench!("histogram","scalar",black_box(scalar::histogram(black_box(&bytes)))[0] as u64);
    bench!("histogram","private_scalar_updates",black_box(scalar::histogram_private(black_box(&bytes)))[0] as u64);
    bench!("adler32","scalar_autovec",u64::from(scalar::adler32(black_box(&bytes))));
    bench!("histogram_score","scalar_libm",math::histogram_score_reference(black_box(&small),&depths,200.0).to_bits());
    bench!("histogram_score","scalar_approx",math::histogram_score_approx(black_box(&small),&depths,200.0,&table).to_bits());
    #[cfg(feature="fearless")]
    {
        use simd_vectorization_recipes::fearless as f;
        let level=fearless_simd::Level::new();
        bench!("histogram_score","fearless_selected_hybrid",f::histogram_score_at(level,black_box(&small),&depths,200.0,&table).to_bits());
        bench!("sum_u32","fearless_public",u64::from(f::sum_wrapping(black_box(&data))));
        bench!("sum_u32","fearless_selected_1acc",u64::from(f::sum_at::<1>(level,black_box(&data))));
        bench!("sum_u32","fearless_selected_2acc",u64::from(f::sum_at::<2>(level,black_box(&data))));
        bench!("sum_u32","fearless_selected_4acc",u64::from(f::sum_at::<4>(level,black_box(&data))));
        bench!("sum_u32","fearless_fixed4_tree",u64::from(f::sum4_at(level,black_box(&data))));
        bench!("product_u32","scalar_autovec",u64::from(scalar::product_wrapping(black_box(&data))));
        bench!("product_u32","fearless_fixed4_tree",u64::from(f::product_at(level,black_box(&data))));
    }
    #[cfg(feature="portable")]
    {
        use simd_vectorization_recipes::portable as p;
        let x:Vec<_>=small.iter().map(|&x| x as f32).collect();
        let rows=vec![[0.5,1.0,-1.0,0.25];n];
        bench!("sum_u32","portable",u64::from(p::sum_wrapping(black_box(&data))));
        bench!("sum_u8","portable_widen16",p::sum_u8_widened(black_box(&bytes)));
        bench!("scan","portable",black_box(p::prefix_sum(black_box(&data))).len() as u64);
        bench!("filter","scalar",black_box(scalar::filter_gt(black_box(&data),u32::MAX/2)).len() as u64);
        bench!("filter","portable_hybrid",black_box(p::filter_gt(black_box(&data),u32::MAX/2)).len() as u64);
        bench!("adler32","portable_block16",u64::from(p::adler32(black_box(&bytes))));
        bench!("batched_dot4","scalar_autovec",u64::from(black_box(scalar::batched_dot4(black_box(&x),&rows))[0].to_bits()));
        bench!("batched_dot4","portable_output_lanes",u64::from(black_box(p::batched_dot4(black_box(&x),&rows))[0].to_bits()));
        bench!("statistics","scalar_autovec",black_box(scalar::statistics_u32(black_box(&data))).sum);
        bench!("statistics","portable_fused",black_box(p::statistics_u32(black_box(&data))).sum);
        bench!("hamming","scalar_autovec",scalar::hamming(black_box(&bytes),black_box(&other_bytes)));
        bench!("hamming","portable_swar",p::hamming(black_box(&bytes),black_box(&other_bytes)));
        bench!("classify_hex","scalar",black_box(scalar::classify_hex(black_box(&bytes))).len() as u64);
        bench!("classify_hex","portable",black_box(p::classify_hex(black_box(&bytes))).len() as u64);
        bench!("histogram_score","portable_hybrid",p::histogram_score(black_box(&small),&depths,200.0,&table).to_bits());
    }
    #[cfg(feature="wide")]
    {
        let floats:Vec<_>=small.iter().map(|&x| (x%8) as f32/8.0).collect();
        bench!("sum_f32_reassociated","scalar_ordered",black_box(&floats).iter().sum::<f32>().to_bits() as u64);
        bench!("sum_f32_reassociated","wide",simd_vectorization_recipes::wide_examples::sum_reassociated(black_box(&floats)).to_bits() as u64);
    }
    Ok(())
}
