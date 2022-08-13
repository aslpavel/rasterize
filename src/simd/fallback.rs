#![allow(non_camel_case_types)]
use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
};

use crate::{linear_to_srgb, srgb_to_linear};

#[derive(Copy, Clone, PartialEq, Default)]
#[repr(transparent)]
pub struct f32x4([f32; 4]);

impl f32x4 {
    #[inline]
    pub fn new(x0: f32, x1: f32, x2: f32, x3: f32) -> Self {
        Self([x0, x1, x2, x3])
    }

    #[inline(always)]
    pub fn x0(self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn x1(self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn x2(self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn x3(self) -> f32 {
        self.0[3]
    }

    #[inline(always)]
    pub fn splat(val: f32) -> Self {
        Self([val, val, val, val])
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        self.into()
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let [x0, x1, x2, x3] = self.0;
        Self([x0.sqrt(), x1.sqrt(), x2.sqrt(), x3.sqrt()])
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

impl Add<Self> for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self::Output {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = other;
        Self([a0 + b0, a1 + b1, a2 + b2, a3 + b3])
    }
}

impl Sub<Self> for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = rhs;
        Self([a0 - b0, a1 - b1, a2 - b2, a3 - b3])
    }
}

impl Mul for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Self) -> Self::Output {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = other;
        Self([a0 * b0, a1 * b1, a2 * b2, a3 * b3])
    }
}

impl Mul<f32> for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, val: f32) -> Self::Output {
        let Self([x0, x1, x2, x3]) = self;
        Self([x0 * val, x1 * val, x2 * val, x3 * val])
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
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = rhs;
        Self([a0 / b0, a1 / b1, a2 / b2, a3 / b3])
    }
}

impl From<[f32; 4]> for f32x4 {
    fn from(arr: [f32; 4]) -> Self {
        Self(arr)
    }
}

impl From<f32x4> for [f32; 4] {
    fn from(val: f32x4) -> Self {
        val.0
    }
}

#[inline(always)]
pub fn l2s(x0: f32x4) -> f32x4 {
    let [r, g, b, a]: [f32; 4] = x0.into();
    f32x4::new(linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b), a)
}

#[inline(always)]
pub fn s2l(x0: f32x4) -> f32x4 {
    let [r, g, b, a]: [f32; 4] = x0.into();
    f32x4::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a)
}
