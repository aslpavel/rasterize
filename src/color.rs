use crate::{Point, Scalar};
use std::{
    fmt,
    ops::{Add, Mul},
    rc::Rc,
    sync::Arc,
};

pub trait Color {
    fn blend_over(&self, other: &Self) -> Self;

    fn with_alpha(&self, alpha: Scalar) -> Self;

    fn to_rgba(&self) -> [u8; 4];

    fn to_rgb(&self) -> [u8; 3]
    where
        Self: Sized,
    {
        let [r, g, b, _] = self.to_rgba();
        [r, g, b]
    }
}

pub trait Paint {
    fn at(&self, point: Point) -> LinColor;
}

impl<'a, P: Paint> Paint for &'a P {
    fn at(&self, point: Point) -> LinColor {
        (**self).at(point)
    }
}

impl Paint for Box<dyn Paint> {
    fn at(&self, point: Point) -> LinColor {
        (**self).at(point)
    }
}

impl Paint for Rc<dyn Paint> {
    fn at(&self, point: Point) -> LinColor {
        (**self).at(point)
    }
}

impl Paint for Arc<dyn Paint> {
    fn at(&self, point: Point) -> LinColor {
        (**self).at(point)
    }
}

/// ABGR color packed as u32 value (most of the platforms are little-endian)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ColorU8(u32);

impl ColorU8 {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(((a as u32) << 24) | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32))
    }

    pub const fn alpha(self) -> u8 {
        ((self.0 >> 24) & 0xff) as u8
    }

    pub const fn blue(self) -> u8 {
        ((self.0 >> 16) & 0xff) as u8
    }

    pub const fn green(self) -> u8 {
        ((self.0 >> 8) & 0xff) as u8
    }

    pub const fn red(self) -> u8 {
        (self.0 & 0xff) as u8
    }

    /*
    fn blend_over(self, other: Self) -> Self {
        // Sa - source alpha
        // Sc - source color
        // Da - destination alpha
        // Dc - destination color
        // Output color would be (prime means premultiplied):
        //   Oc' = Sc * Sa + Dc * Da * (1 - Sa)
        //   Oa = Sa + Da * (1 - Sa)
        //   Oc = Oc' / Oa
        //
        //   Oc' = lerp(Dc * Da, Sc, Sa)
        //   Oa = Sa + Da - Sa * Da
        let da = u32::from(self.alpha());
        let sa = u32::from(other.alpha());
        let _oa = sa + da - mul_u8(sa, da);

        let dc = self.0;
        let sc = other.0;
        let _oc = lerp_u8x4(mul_u8x4(dc, da), sc, sa);
        todo!()
    }
    */
}

impl Color for ColorU8 {
    fn to_rgba(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    fn blend_over(&self, _other: &Self) -> Self {
        todo!()
    }

    fn with_alpha(&self, _alpha: Scalar) -> Self {
        todo!()
    }
}

impl From<LinColor> for ColorU8 {
    fn from(lin: LinColor) -> Self {
        let LinColor([r, g, b, a]) = lin;
        if a < std::f32::EPSILON {
            return ColorU8::default();
        }
        let r = (linear_to_srgb(r / a) * 255.0 + 0.5) as u8;
        let g = (linear_to_srgb(g / a) * 255.0 + 0.5) as u8;
        let b = (linear_to_srgb(b / a) * 255.0 + 0.5) as u8;
        let a = (a * 255.0 + 0.5) as u8;
        ColorU8::new(r, g, b, a)
    }
}

impl fmt::Debug for ColorU8 {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("ColorU8")
            .field("r", &self.red())
            .field("g", &self.green())
            .field("b", &self.blue())
            .field("a", &self.alpha())
            .finish()
    }
}

/// Alpha premultiplied RGBA color in the liniar color space (no gamma correction)
#[derive(Clone, Copy, PartialEq, Default)]
pub struct LinColor([f32; 4]);

impl LinColor {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        LinColor([r, g, b, a])
    }

    pub fn red(&self) -> f32 {
        self.0[0]
    }

    pub fn green(&self) -> f32 {
        self.0[1]
    }

    pub fn blue(&self) -> f32 {
        self.0[2]
    }

    pub fn alpha(&self) -> f32 {
        self.0[3]
    }
}

impl Color for LinColor {
    fn to_rgba(&self) -> [u8; 4] {
        ColorU8::from(*self).to_rgba()
    }

    fn blend_over(&self, other: &Self) -> Self {
        *other + *self * (1.0 - other.alpha())
    }

    fn with_alpha(&self, alpha: Scalar) -> Self {
        *self * (alpha as f32)
    }
}

impl Paint for LinColor {
    fn at(&self, _: Point) -> LinColor {
        *self
    }
}

impl Add<Self> for LinColor {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let Self([r0, g0, b0, a0]) = self;
        let Self([r1, g1, b1, a1]) = other;
        Self([r0 + r1, g0 + g1, b0 + b1, a0 + a1])
    }
}

impl Mul<f32> for LinColor {
    type Output = Self;

    #[inline]
    fn mul(self, scale: f32) -> Self::Output {
        let Self([r, g, b, a]) = self;
        Self([scale * r, scale * g, scale * b, scale * a])
    }
}

impl Color for Scalar {
    fn to_rgba(&self) -> [u8; 4] {
        let color = (linear_to_srgb(1.0 - (*self as f32)) * 255.0 + 0.5) as u8;
        [color, color, color, 255]
    }

    fn blend_over(&self, other: &Self) -> Self {
        other + self * (1.0 - other)
    }

    fn with_alpha(&self, alpha: Scalar) -> Self {
        self * alpha
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/*
const MASK_LOW: u32 = 0x00FF00FF;

/// Calculate `[_, a1, _, a3] * b / 255`, where `a{0-3}` and `b` are `u8`
///
/// Those optimisation comes from for these two formulas:
///   1. `a + ar + ar^2 ... = a / (1 - r)` for all r in [0..1)
///   2. `t / 255 = (t / 256) / (1 - r)` where if `r = 1 / 256`
///
/// Applying this 1 and 2, takin only first two arguments from 1.
///   `(v >> 8) + (v >> 16)` => `((v >> 8) + v) >> 8` where `v = a * b`
///
/// Basically we get `v / 255 = ((v >> 8) + v) >> 8`
///
/// This function also doing this operation at on two u8 at once by means of masking
/// and then recomposing everyting in one value.
///
/// References:
///   - [Image Compasiting Fundamentals](https://www.cs.princeton.edu/courses/archive/fall00/cs426/papers/smith95a.pdf)
///   - [Double blend trick](http://stereopsis.com/doubleblend.html)
fn mul_u8x2(a: u32, b: u32) -> u32 {
    let m0 = (a & MASK_LOW) * b + 0x00800080;
    ((((m0 >> 8) & MASK_LOW) + m0) >> 8) & MASK_LOW
}

fn mul_u8(a: u32, b: u32) -> u32 {
    let t = (a * b) + 0x80;
    ((t >> 8) + t) >> 8
}

pub fn mul_u8x4(a: u32, b: u32) -> u32 {
    let low = mul_u8x2(a, b);
    let high = mul_u8x2(a >> 8, b) << 8;
    low + high
}

fn lerp_u8x2(a: u32, b: u32, t: u32) -> u32 {
    let a_low = a & MASK_LOW;
    let b_low = b & MASK_LOW;
    let delta_low = b_low.wrapping_sub(a_low).wrapping_mul(t) >> 8;
    (a_low + delta_low) & MASK_LOW
}

pub fn lerp_u8x4(a: u32, b: u32, t: u32) -> u32 {
    let low = lerp_u8x2(a, b, t);
    let high = lerp_u8x2(a >> 8, b >> 8, t) << 8;
    low | high
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_u8() {
        let c = ColorU8::new(1, 2, 3, 4);
        assert_eq!([1, 2, 3, 4], c.to_rgba());
        assert_eq!(1, c.red());
        assert_eq!(2, c.green());
        assert_eq!(3, c.blue());
        assert_eq!(4, c.alpha());
    }

    /*
    #[test]
    fn test_mul_u8x4() {
        assert_eq!(mul_u8x4(0xff804020, 0x20), 0x20100804);
    }
    */
}
