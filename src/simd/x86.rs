#![allow(non_camel_case_types)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::{
    fmt,
    ops::{Add, Mul},
};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct f32x4(__m128);

impl f32x4 {
    #[inline(always)]
    pub fn new(x0: f32, x1: f32, x2: f32, x3: f32) -> Self {
        Self(unsafe { _mm_set_ps(x3, x2, x1, x0) })
    }

    #[inline(always)]
    pub fn splat(val: f32) -> Self {
        Self(unsafe { _mm_set1_ps(val) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm_setzero_ps() })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        self.into()
    }
}

impl fmt::Debug for f32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [x0, x1, x2, x3] = self.to_array();
        f.debug_tuple("f32x4")
            .field(&x0)
            .field(&x1)
            .field(&x2)
            .field(&x3)
            .finish()
    }
}

impl Default for f32x4 {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for f32x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        let mask = unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.0, other.0)) };
        mask == 0b1111
    }
}

impl Add<Self> for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self::Output {
        Self(unsafe { _mm_add_ps(self.0, other.0) })
    }
}

impl Mul for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Self) -> Self::Output {
        Self(unsafe { _mm_mul_ps(self.0, other.0) })
    }
}

impl Mul<f32> for f32x4 {
    type Output = Self;

    // #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        self * Self::splat(rhs)
    }
}

impl From<[f32; 4]> for f32x4 {
    #[inline(always)]
    fn from(arr: [f32; 4]) -> Self {
        // Safety: because this semantically moves the value from the input position
        // (align4) to the output position (align16) it is fine to increase our
        // required alignment without worry.
        unsafe { core::mem::transmute(arr) }
    }
}

impl From<f32x4> for [f32; 4] {
    #[inline(always)]
    fn from(m: f32x4) -> Self {
        // We can of course transmute to a lower alignment
        unsafe { core::mem::transmute(m) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        unsafe {
            println!("{:?}", _mm_shuffle_ps::<0b00110011>(a.0, b.0));
        }
    }
}
