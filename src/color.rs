use crate::{simd::f32x4, Paint, Point, Scalar, Transform, Units};
use std::{
    fmt,
    ops::{Add, Mul},
    str::FromStr,
};

/// Common interface to all color representations
pub trait Color: Copy {
    /// Blend other color on top of this color
    fn blend_over(self, other: Self) -> Self;

    /// Override alpha component of the color
    fn with_alpha(self, alpha: Scalar) -> Self;

    /// Convert color to sRGBA list
    fn to_rgba(self) -> [u8; 4];

    /// Convert color to sRGB list (alpha is discarded)
    fn to_rgb(self) -> [u8; 3]
    where
        Self: Sized,
    {
        let [r, g, b, _] = self.to_rgba();
        [r, g, b]
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
}

impl Color for ColorU8 {
    fn to_rgba(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    fn blend_over(self, _other: Self) -> Self {
        todo!()
    }

    fn with_alpha(self, _alpha: Scalar) -> Self {
        todo!()
    }
}

impl From<LinColor> for ColorU8 {
    fn from(lin: LinColor) -> Self {
        let [r, g, b, a]: [f32; 4] = lin.into();
        if a <= std::f32::EPSILON {
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

impl FromStr for ColorU8 {
    type Err = ColorError;

    fn from_str(color: &str) -> Result<Self, Self::Err> {
        if color.starts_with('#') && (color.len() == 7 || color.len() == 9) {
            // #RRGGBB(AA)
            let bytes: &[u8] = color[1..].as_ref();
            let digit = |byte| match byte {
                b'A'..=b'F' => Ok(byte - b'A' + 10),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                b'0'..=b'9' => Ok(byte - b'0'),
                _ => Err(ColorError::HexExpected),
            };
            let mut hex = bytes
                .chunks(2)
                .map(|pair| Ok(digit(pair[0])? << 4 | digit(pair[1])?));
            Ok(ColorU8::new(
                hex.next().unwrap_or(Ok(0))?,
                hex.next().unwrap_or(Ok(0))?,
                hex.next().unwrap_or(Ok(0))?,
                hex.next().unwrap_or(Ok(255))?,
            ))
        } else {
            Err(ColorError::HexExpected)
        }
    }
}

impl fmt::Display for ColorU8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [r, g, b, a] = self.to_rgba();
        write!(f, "#{:02x}{:02x}{:02x}", r, g, b)?;
        if a != 255 {
            write!(f, "{:02x}", a)?;
        }
        Ok(())
    }
}

/// Alpha premultiplied RGBA color in the linear color space (no gamma correction)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LinColor(crate::simd::f32x4);

impl LinColor {
    #[inline(always)]
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        LinColor(f32x4::new(r, g, b, a))
    }

    #[inline(always)]
    pub fn red(self) -> f32 {
        self.0.x0()
    }

    #[inline(always)]
    pub fn green(self) -> f32 {
        self.0.x1()
    }

    #[inline(always)]
    pub fn blue(self) -> f32 {
        self.0.x2()
    }

    #[inline(always)]
    pub fn alpha(self) -> f32 {
        self.0.x3()
    }

    #[inline(always)]
    pub fn lerp(self, other: LinColor, t: Scalar) -> Self {
        let t = t as f32;
        other * t + self * (1.0 - t)
    }

    /// Convert into alpha-premultiplied SRGB from Linear RGB
    ///
    /// Used by gradients, do not make public
    #[inline(always)]
    pub(crate) fn into_srgb(self) -> Self {
        // !!! check firefox scene
        // let alpha = f32x4::splat(self.alpha());
        // Self(l2s(self.0 / alpha) * alpha)

        let [r, g, b, a]: [f32; 4] = self.into();
        if a <= 1e-6 {
            Self::new(0.0, 0.0, 0.0, 0.0)
        } else {
            Self::new(
                linear_to_srgb(r / a) * a,
                linear_to_srgb(g / a) * a,
                linear_to_srgb(b / a) * a,
                a,
            )
        }
    }

    /// Convert into alpha-premultiplied Linear RGB from SRGB
    ///
    /// Used by gradient, do not make public
    #[inline(always)]
    pub(crate) fn into_linear(self) -> Self {
        let [r, g, b, a]: [f32; 4] = self.into();
        if a <= 1e-6 {
            return Self::new(0.0, 0.0, 0.0, 0.0);
        }
        Self::new(
            srgb_to_linear(r / a) * a,
            srgb_to_linear(g / a) * a,
            srgb_to_linear(b / a) * a,
            a,
        )
    }
}

impl Color for LinColor {
    #[inline(always)]
    fn to_rgba(self) -> [u8; 4] {
        ColorU8::from(self).to_rgba()
    }

    #[inline(always)]
    fn blend_over(self, other: Self) -> Self {
        other + self * (1.0 - other.alpha())
    }

    #[inline(always)]
    fn with_alpha(self, alpha: Scalar) -> Self {
        self * (alpha as f32)
    }
}

impl Paint for LinColor {
    fn at(&self, _: Point) -> LinColor {
        *self
    }

    fn units(&self) -> Option<Units> {
        None
    }

    fn transform(&self) -> Transform {
        Transform::identity()
    }

    #[cfg(feature = "serde")]
    fn to_json(&self) -> Result<serde_json::Value, crate::SvgParserError> {
        Ok(serde_json::Value::String(self.to_string()))
    }
}

impl Add<Self> for LinColor {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}

impl Mul<f32> for LinColor {
    type Output = Self;

    #[inline(always)]
    fn mul(self, scale: f32) -> Self::Output {
        Self(self.0 * scale)
    }
}

impl From<ColorU8> for LinColor {
    fn from(color: ColorU8) -> Self {
        // !!! overshots at 1.0
        // let [r, g, b, a] = color.to_rgba();
        // let rgba = f32x4::new(r as f32, g as f32, b as f32, 255.0) * 0.00392156862745098;
        // LinColor(s2l(rgba) * f32x4::splat(a as f32 / 255.0))
        let a = color.alpha() as f32 / 255.0;
        let r = srgb_to_linear(color.red() as f32 / 255.0) * a;
        let g = srgb_to_linear(color.green() as f32 / 255.0) * a;
        let b = srgb_to_linear(color.blue() as f32 / 255.0) * a;
        LinColor::new(r, g, b, a)
    }
}

impl From<LinColor> for [f32; 4] {
    fn from(color: LinColor) -> Self {
        color.0.into()
    }
}

impl FromStr for LinColor {
    type Err = ColorError;

    fn from_str(color: &str) -> Result<Self, Self::Err> {
        Ok(ColorU8::from_str(color)?.into())
    }
}

impl fmt::Display for LinColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ColorU8::from(*self).fmt(f)
    }
}

impl Color for Scalar {
    fn to_rgba(self) -> [u8; 4] {
        let color = (linear_to_srgb(1.0 - (self as f32)) * 255.0 + 0.5) as u8;
        [color, color, color, 255]
    }

    fn blend_over(self, other: Self) -> Self {
        other + self * (1.0 - other)
    }

    fn with_alpha(self, alpha: Scalar) -> Self {
        self * alpha
    }
}

/// Convert Linear RGB color component into a SRGB color component.
///
/// It was hard to optimize this function, even current version
/// is slow because of the conditional jump. Lookup table is not working
/// here as well it should be at least 4K in size an not cache friendly.
///
/// Precise implementation
/// ```no_run
/// pub fn linear_to_srgb(value: f32) -> f32 {
///     if value <= 0.0031308 {
///         value * 12.92
///     } else {
///         1.055 * value.powf(1.0 / 2.4) - 0.055
///     }
/// }
/// ```
#[inline]
pub fn linear_to_srgb(x0: f32) -> f32 {
    if x0 <= 0.0031308 {
        x0 * 12.92
    } else {
        // This function is generated by least square fitting of
        // `f(x) = 1.055 * x.powf(1.0 / 2.4) - 0.055` on value [0.0031308..1.0]
        // see `scripts/srgb.py` for details.
        let x1 = x0.sqrt();
        let x2 = x1.sqrt();
        let x3 = x2.sqrt();
        -0.01848558 * x0 + 0.6445592 * x1 + 0.70994765 * x2 - 0.33605254 * x3
    }
}

#[inline]
pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[derive(Debug, Clone)]
pub enum ColorError {
    HexExpected,
}

impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColorError::HexExpected => {
                write!(f, "Color expected to be #RRGGBB(AA) in hexidemical format")
            }
        }
    }
}

impl std::error::Error for ColorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    fn test_color_u8() {
        let c = ColorU8::new(1, 2, 3, 4);
        assert_eq!([1, 2, 3, 4], c.to_rgba());
        assert_eq!(1, c.red());
        assert_eq!(2, c.green());
        assert_eq!(3, c.blue());
        assert_eq!(4, c.alpha());
    }

    #[test]
    fn test_color_u8_parse() -> Result<(), ColorError> {
        assert_eq!(ColorU8::new(1, 2, 3, 4), "#01020304".parse::<ColorU8>()?);
        assert_eq!(
            ColorU8::new(170, 187, 204, 255),
            "#aabbcc".parse::<ColorU8>()?
        );
        assert_eq!(ColorU8::new(0, 0, 0, 255), "#000000".parse::<ColorU8>()?);
        Ok(())
    }

    #[test]
    fn test_conversion() -> Result<(), ColorError> {
        let c: ColorU8 = "#ff804010".parse()?;
        let l: LinColor = c.into();
        let r: ColorU8 = l.into();
        assert_eq!(c, r);
        Ok(())
    }

    #[test]
    fn test_lin_and_srgb() {
        for i in 0..255 {
            let v = i as f32 / 255.0;
            assert_approx_eq!(v, linear_to_srgb(srgb_to_linear(v)), 1e-4);
            assert_approx_eq!(v, srgb_to_linear(linear_to_srgb(v)), 1e-4);
        }
    }

    #[test]
    fn test_display_parse() -> Result<(), ColorError> {
        let c: ColorU8 = "#01020304".parse()?;
        assert_eq!(c, ColorU8::new(1, 2, 3, 4));
        assert_eq!(c.to_string(), "#01020304");

        let c: ColorU8 = "#010203".parse()?;
        assert_eq!(c, ColorU8::new(1, 2, 3, 255));
        assert_eq!(c.to_string(), "#010203");

        Ok(())
    }

    /*
    #[test]
    fn test_mul_u8x4() {
        assert_eq!(mul_u8x4(0xff804020, 0x20), 0x20100804);
    }
    */
}
