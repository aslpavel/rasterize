#![allow(non_camel_case_types)]

use bytemuck::{Pod, Zeroable};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::{
    fmt,
    mem::transmute,
    ops::{Add, Div, Mul, Sub},
};

#[repr(transparent)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct f32x4(__m128);

impl f32x4 {
    #[inline(always)]
    pub fn new(x0: f32, x1: f32, x2: f32, x3: f32) -> Self {
        Self(unsafe { _mm_set_ps(x3, x2, x1, x0) })
    }

    pub fn fallback(self) -> super::fallback::f32x4 {
        let this: [f32; 4] = self.into();
        this.into()
    }

    #[inline(always)]
    pub fn x0(self) -> f32 {
        f32::from_bits(unsafe { _mm_extract_ps::<0>(self.0) } as u32)
    }

    #[inline(always)]
    pub fn x1(self) -> f32 {
        f32::from_bits(unsafe { _mm_extract_ps::<1>(self.0) } as u32)
    }

    #[inline(always)]
    pub fn x2(self) -> f32 {
        f32::from_bits(unsafe { _mm_extract_ps::<2>(self.0) } as u32)
    }

    #[inline(always)]
    pub fn x3(self) -> f32 {
        f32::from_bits(unsafe { _mm_extract_ps::<3>(self.0) } as u32)
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

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm_sqrt_ps(self.0) })
    }

    #[inline(always)]
    pub fn mul_add(self, mul: f32x4, add: f32x4) -> Self {
        Self(unsafe { _mm_fmadd_ps(self.0, mul.0, add.0) })
    }

    #[inline(always)]
    pub fn add_mul(self, a: f32x4, b: f32x4) -> Self {
        a.mul_add(b, self)
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> f32 {
        let result = unsafe { _mm_extract_ps::<0>(_mm_dp_ps::<0b1111_1111>(self.0, other.0)) };
        f32::from_bits(result as u32)
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

impl Sub<Self> for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_sub_ps(self.0, rhs.0) })
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

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        self * Self::splat(rhs)
    }
}

impl Mul<f32x4> for f32 {
    type Output = f32x4;

    #[inline(always)]
    fn mul(self, rhs: f32x4) -> Self::Output {
        rhs * f32x4::splat(self)
    }
}

impl Div<f32x4> for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: f32x4) -> Self::Output {
        Self(unsafe { _mm_div_ps(self.0, rhs.0) })
    }
}

impl From<[f32; 4]> for f32x4 {
    #[inline(always)]
    fn from(arr: [f32; 4]) -> Self {
        // Safety: because this semantically moves the value from the input position
        // (align4) to the output position (align16) it is fine to increase our
        // required alignment without worry.
        unsafe { transmute(arr) }
    }
}

impl From<f32x4> for [f32; 4] {
    #[inline(always)]
    fn from(m: f32x4) -> Self {
        // We can of course transmute to a lower alignment
        unsafe { transmute(m) }
    }
}

impl IntoIterator for f32x4 {
    type Item = f32;
    type IntoIter = <[f32; 4] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        let vals: [f32; 4] = self.into();
        vals.into_iter()
    }
}

#[inline(always)]
pub fn l2s(x0: f32x4) -> f32x4 {
    let x1 = x0.sqrt();
    let x2 = x1.sqrt();
    let x3 = x2.sqrt();
    let high = -0.01848558 * x0 + 0.6445592 * x1 + 0.70994765 * x2 - 0.33605254 * x3;
    // much slower without `-C target-cpu=native`
    // let high = (-0.01848558 * x0)
    //     .add_mul(f32x4::splat(0.6445592), x1)
    //     .add_mul(f32x4::splat(0.70994765), x2)
    //     .add_mul(f32x4::splat(-0.33605254), x3);
    unsafe {
        f32x4(_mm_blendv_ps(
            high.0,
            (x0 * 12.92).0,
            _mm_cmple_ps(x0.0, _mm_set1_ps(0.0031308)),
        ))
    }
}

#[inline(always)]
pub fn s2l(vs: f32x4) -> f32x4 {
    // def s2l(value):
    //   if value <= 0.04045:
    //     return value / 12.92
    //   else:
    //     return ((value + 0.055) / 1.055) ** 2.4
    // x = np.linspace(0.04045, 1, 16000)
    // y = np.array([s2l(v) for v in x])
    // np.polynomial.Polynomial.fit(x, y, 3)
    // 𝑥 ↦ 0.23361048543711943 +
    //      0.4665843122387033 * (-1.0843103538116827 + 2.0843103538116825 * 𝑥) +
    //      0.26901741378006355 * (-1.0843103538116827 + 2.0843103538116825 * 𝑥) ^2 +
    //      0.031661580753065945 * (-1.0843103538116827 + 2.0843103538116825 * 𝑥) ^ 3
    let x1 = 2.0843103538116825 * vs - f32x4::splat(1.0843103538116827);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let vs_high = f32x4::splat(0.23361048543711943)
        + 0.4665843122387033 * x1
        + 0.26901741378006355 * x2
        + 0.031661580753065945 * x3;
    unsafe {
        f32x4(_mm_blendv_ps(
            vs_high.0,
            (vs * 0.07739938080495357).0,
            _mm_cmple_ps(vs.0, _mm_set1_ps(0.04045)),
        ))
    }
}

/// Create shuffle/permute mask
pub const fn shuffle_mask(x0: u32, x1: u32, x2: u32, x3: u32) -> i32 {
    ((x3 << 6) | (x2 << 4) | (x1 << 2) | x0) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a.x0(), 1.0);
        assert_eq!(a.x1(), 2.0);
        assert_eq!(a.x2(), 3.0);
        assert_eq!(a.x3(), 4.0);

        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        unsafe {
            assert_eq!(
                f32x4(_mm_shuffle_ps::<{ shuffle_mask(0, 3, 0, 3) }>(a.0, b.0)),
                f32x4::new(1.0, 4.0, 5.0, 8.0),
            );
            assert_eq!(
                f32x4(_mm_permute_ps::<{ shuffle_mask(0, 3, 3, 2) }>(b.0)),
                f32x4::new(5.0, 8.0, 8.0, 7.0)
            );
        }

        let c = f32x4::new(0.001, 0.1, 0.2, 0.7);
        println!("{c:?}");
        println!("{:?}", l2s(c));
        println!("{:?}", s2l(l2s(c)));
        dbg!(s2l(f32x4::splat(1.0)));
    }

    #[test]
    fn test_dot() {
        let a = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = f32x4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(70.0, a.dot(b));
        assert_eq!(70.0, a.fallback().dot(b.fallback()));
    }
}
