//! Linux-only physical-end boundary test. Not a sanitizer replacement.
#![cfg(target_os = "linux")]
use std::{ffi::{c_int,c_long,c_void}, ptr};
use simd_vectorization_recipes::{arch,scalar};
extern "C" {
    fn sysconf(name:c_int)->c_long;
    fn mmap(addr:*mut c_void,len:usize,prot:c_int,flags:c_int,fd:c_int,offset:isize)->*mut c_void;
    fn mprotect(addr:*mut c_void,len:usize,prot:c_int)->c_int;
    fn munmap(addr:*mut c_void,len:usize)->c_int;
}
struct Mapping { base:*mut u8, page:usize }
impl Mapping {
    fn new()->Self {
        // Linux ABI constants: _SC_PAGESIZE=30; PROT_READ|WRITE=3;
        // MAP_PRIVATE|MAP_ANONYMOUS=0x22. Test is Linux-only.
        unsafe {
            let page=sysconf(30); assert!(page>0);
            let page=page as usize;
            let base=mmap(ptr::null_mut(),2*page,3,0x22,-1,0);
            assert_ne!(base,(-1isize) as *mut c_void,"mmap failed");
            assert!(!base.is_null(),"a null mapping cannot back a Rust slice");
            let result=Self {base:base.cast(),page};
            assert_eq!(mprotect(result.base.add(page).cast(),page,0),0,"mprotect failed");
            result
        }
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this object owns exactly this still-mapped two-page allocation.
        unsafe { let _=munmap(self.base.cast(),2*self.page); }
    }
}
#[test]
fn arbitrary_slice_ends_need_no_padding() {
    let region=Mapping::new();
    for n in 0..=257.min(region.page/4) {
        // SAFETY: first page is writable; n u32s fit and end at page boundary;
        // page-size alignment and multiples of four preserve u32 alignment.
        unsafe {
            let start=region.base.add(region.page-4*n).cast::<u32>();
            for i in 0..n { start.add(i).write((i as u32).wrapping_mul(12345)); }
            let x=std::slice::from_raw_parts(start,n);
            for backend in arch::supported_backends() {
                assert_eq!(arch::sum_with(backend,x),scalar::sum_wrapping(x));
                assert_eq!(arch::prefix_with(backend,x),scalar::prefix_sum(x));
            }
            #[cfg(feature="fearless")]
            assert_eq!(simd_vectorization_recipes::fearless::sum_wrapping(x),scalar::sum_wrapping(x));
            #[cfg(feature="portable")]
            assert_eq!(simd_vectorization_recipes::portable::sum_wrapping(x),scalar::sum_wrapping(x));
        }
    }
    for n in 0..=257.min(region.page) {
        // SAFETY: exactly n bytes in the writable page are initialized; the next
        // byte belongs to a PROT_NONE page. Slices remain borrowed within scope.
        unsafe {
            let start=region.base.add(region.page-n);
            ptr::write_bytes(start,7,n);
            let x=std::slice::from_raw_parts(start,n);
            for backend in arch::supported_backends() {
                assert_eq!(arch::sum_u8_with(backend,x),7*(n as u64));
                assert_eq!(arch::find_with(backend,x,99),None);
            }
        }
    }
}
