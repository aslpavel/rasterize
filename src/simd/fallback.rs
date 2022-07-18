#![allow(non_camel_case_types)]
use std::{
    fmt,
    ops::{Add, Mul},
};

#[derive(Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct f32x4([f32; 4]);

impl f32x4 {
    #[inline]
    pub fn new(x0: f32, x1: f32, x2: f32, x3: f32) -> Self {
        Self([x0, x1, x2, x3])
    }

    #[inline]
    pub fn splat(val: f32) -> Self {
        Self([val, val, val, val])
    }

    #[inline]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

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

impl Add<Self> for f32x4 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = other;
        Self([a0 + b0, a1 + b1, a2 + b2, a3 + b3])
    }
}

impl Mul for f32x4 {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = other;
        Self([a0 * b0, a1 * b1, a2 * b2, a3 * b3])
    }
}

impl Mul<f32> for f32x4 {
    type Output = Self;

    #[inline]
    fn mul(self, val: f32) -> Self::Output {
        let Self([x0, x1, x2, x3]) = self;
        Self([x0 * val, x1 * val, x2 * val, x3 * val])
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
