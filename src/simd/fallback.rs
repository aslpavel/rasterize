use std::ops::{Add, Mul};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct f32x4 {
    x0: f32,
    x1: f32,
    x2: f32,
    x3: f32,
}

impl f32x4 {
    #[inline]
    pub fn new(x0: f32, x1: f32, x2: f32, x3: f32) -> Self {
        Self { x0, x1, x2, x3 }
    }

    #[inline]
    pub fn zero() -> Self {
        Self {
            x0: 0.0,
            x1: 0.0,
            x2: 0.0,
            x3: 0.0,
        }
    }
}

impl Add<Self> for f32x4 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        Self {
            x0: self.x0 + other.x0,
            x1: self.x1 + other.x1,
            x2: self.x2 + other.x2,
            x3: self.x3 + other.x3,
        }
    }
}

impl Mul<f32> for f32x4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x0: self.x0 * rhs,
            x1: self.x1 * rhs,
            x2: self.x2 * rhs,
            x3: self.x3 * rhs,
        }
    }
}
