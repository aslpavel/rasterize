pub mod fallback;
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd")))]
pub use fallback::*;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
mod x86;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
pub use x86::*;
