#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Line, Size, SvgParserError, utils::clamp};
use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
    str::FromStr,
};

/// Scalar type
pub type Scalar = f64;
/// Epsilon value
pub const EPSILON: f64 = f64::EPSILON;
/// Square root of the epsilon value
pub const EPSILON_SQRT: f64 = 1.490_116_119_384_765_6e-8;
/// Mathematical pi constant
pub const PI: f64 = std::f64::consts::PI;

const SCALAR_PRECISION: usize = 4;
const SCALAR_FORMAT: u128 = lexical_core::NumberFormatBuilder::new()
    .required_integer_digits(false)
    .build();
const SCALAR_FORMAT_OPTIONS: lexical_core::WriteFloatOptions =
    lexical_core::WriteFloatOptions::builder()
        .max_significant_digits(std::num::NonZero::new(SCALAR_PRECISION))
        .trim_floats(true)
        .build_unchecked();

pub struct ScalarFormatter {
    precision: usize,
    round: bool, // whether to preround (correctly removes)
    options: lexical_core::WriteFloatOptions,
    buffer: [u8; lexical_core::BUFFER_SIZE],
}
impl ScalarFormatter {
    pub fn new(precision: Option<usize>, round: bool) -> Self {
        let options = precision
            .and_then(|precision| {
                lexical_core::WriteFloatOptionsBuilder::new()
                    .max_significant_digits(std::num::NonZero::new(precision))
                    .trim_floats(true)
                    .build()
                    .ok()
            })
            .unwrap_or(SCALAR_FORMAT_OPTIONS);
        Self {
            precision: precision.unwrap_or(SCALAR_PRECISION),
            options,
            round,
            buffer: [0u8; lexical_core::BUFFER_SIZE],
        }
    }

    pub fn new_fmt(fmt: &fmt::Formatter<'_>) -> Self {
        Self::new(fmt.precision(), fmt.alternate())
    }

    pub fn format(&mut self, mut value: Scalar) -> &[u8] {
        if self.round {
            value = Self::round_significant(value, self.precision);
        }
        lexical_core::write_with_options::<_, SCALAR_FORMAT>(value, &mut self.buffer, &self.options)
    }

    pub fn format_str(&mut self, value: Scalar) -> &str {
        unsafe {
            // SAFETY: trust lexical to produce valid utf-8 string
            std::str::from_utf8_unchecked(self.format(value))
        }
    }

    pub fn round_significant(value: f64, precision: usize) -> f64 {
        if value.abs() < EPSILON {
            0.0
        } else {
            let shift = precision as i32 - value.abs().log10().ceil() as i32;
            let shift_factor = 10_f64.powi(shift);
            (value * shift_factor).round() / shift_factor
        }
    }
}

pub struct ScalarFormat(pub Scalar);

impl fmt::Debug for ScalarFormat {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatter = ScalarFormatter::new_fmt(fmt);
        fmt.write_str(formatter.format_str(self.0))
    }
}

impl fmt::Display for ScalarFormat {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatter = ScalarFormatter::new_fmt(fmt);
        fmt.write_str(formatter.format_str(self.0))
    }
}

/// Value representing a 2D point or vector.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct Point(pub [Scalar; 2]);

#[allow(clippy::derived_hash_with_manual_eq)]
impl std::hash::Hash for Point {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let Point([x, y]) = self;
        x.to_be_bytes().hash(state);
        y.to_be_bytes().hash(state);
    }
}

impl fmt::Debug for Point {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Point([x, y]) = self;
        let mut formatter = ScalarFormatter::new_fmt(fmt);
        fmt.write_str(formatter.format_str(*x))?;
        fmt.write_str(",")?;
        fmt.write_str(formatter.format_str(*y))?;
        Ok(())
    }
}

impl Point {
    #[inline]
    pub const fn new(x: Scalar, y: Scalar) -> Self {
        Self([x, y])
    }

    /// Get `x` component of the point
    #[inline]
    pub const fn x(self) -> Scalar {
        self.0[0]
    }

    /// Get `y` component of the point
    #[inline]
    pub const fn y(self) -> Scalar {
        self.0[1]
    }

    /// Get length of the vector (distance from the origin)
    pub fn length(self) -> Scalar {
        let Self([x, y]) = self;
        x.hypot(y)
    }

    /// Distance between two points
    pub fn dist(self, other: Self) -> Scalar {
        (self - other).length()
    }

    /// Dot product between two vectors
    pub fn dot(self, other: Self) -> Scalar {
        let Self([x0, y0]) = self;
        let Self([x1, y1]) = other;
        x0 * x1 + y0 * y1
    }

    /// Cross product between two vectors
    pub fn cross(self, other: Self) -> Scalar {
        let Self([x0, y0]) = self;
        let Self([x1, y1]) = other;
        x0 * y1 - y0 * x1
    }

    /// Get vector normal (not a unit sized)
    pub fn normal(self) -> Point {
        let Self([x, y]) = self;
        Self([y, -x])
    }

    /// Convert vector to a unit size vector, if length is not zero
    pub fn normalize(self) -> Option<Point> {
        let Self([x, y]) = self;
        let length = self.length();
        if length < EPSILON {
            None
        } else {
            Some(Self([x / length, y / length]))
        }
    }

    /// Calculate angle (from self to the other) between two vectors
    pub fn angle_between(self, other: Self) -> Option<Scalar> {
        let angle = clamp(self.cos_between(other)?, -1.0, 1.0).acos();
        if self.cross(other) < 0.0 {
            Some(-angle)
        } else {
            Some(angle)
        }
    }

    /// Cosine of the angle (from self to the other) between to vectors
    pub fn cos_between(self, other: Self) -> Option<Scalar> {
        let lengths = self.length() * other.length();
        if lengths < EPSILON {
            None
        } else {
            Some(self.dot(other) / lengths)
        }
    }

    /// Sine of the angle (from self to the other) between to vectors
    pub fn sin_between(self, other: Self) -> Option<Scalar> {
        let cos = self.cos_between(other)?;
        Some((1.0 - cos * cos).sqrt())
    }

    /// Determine if self is close to the other within the margin of error (EPSILON)
    pub fn is_close_to(self, other: Point) -> bool {
        let Self([x0, y0]) = self;
        let Self([x1, y1]) = other;
        (x0 - x1).abs() < EPSILON && (y0 - y1).abs() < EPSILON
    }
}

impl From<(Scalar, Scalar)> for Point {
    #[inline]
    fn from(xy: (Scalar, Scalar)) -> Self {
        Self([xy.0, xy.1])
    }
}

impl<'a> From<&'a Point> for Point {
    fn from(point: &'a Point) -> Self {
        let Self([x, y]) = point;
        Self([*x, *y])
    }
}

impl Mul<&Point> for Scalar {
    type Output = Point;

    #[inline]
    fn mul(self, other: &Point) -> Self::Output {
        let Point([x, y]) = other;
        Point([self * x, self * y])
    }
}

impl Mul<Point> for Scalar {
    type Output = Point;

    #[inline]
    fn mul(self, other: Point) -> Self::Output {
        let Point([x, y]) = other;
        Point([self * x, self * y])
    }
}

impl Div<Scalar> for Point {
    type Output = Point;

    #[inline]
    fn div(self, rhs: Scalar) -> Self::Output {
        let Point([x, y]) = self;
        Point([x / rhs, y / rhs])
    }
}

impl Add for Point {
    type Output = Point;

    #[inline]
    fn add(self, other: Point) -> Self::Output {
        let Point([x0, y0]) = self;
        let Point([x1, y1]) = other;
        Point([x0 + x1, y0 + y1])
    }
}

impl Sub for Point {
    type Output = Point;

    #[inline]
    fn sub(self, other: Point) -> Self::Output {
        let Point([x0, y0]) = self;
        let Point([x1, y1]) = other;
        Point([x0 - x1, y0 - y1])
    }
}

impl Mul for Point {
    type Output = Point;

    #[inline]
    fn mul(self, other: Point) -> Self::Output {
        let Point([x0, y0]) = self;
        let Point([x1, y1]) = other;
        Point([x0 * x1, y0 * y1])
    }
}

/// Alignment options
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Align {
    /// Align by minimal value
    Min,
    /// Align by center value
    Mid,
    /// Align by maximum value
    Max,
}

/// 2D affine transformation
///
/// Stored as an array [m00, m01, m02, m10, m11, m12], semantically corresponds to
/// a matrix:
/// ┌             ┐
/// │ m00 m01 m02 │
/// │ m11 m11 m12 │
/// │   0   0   1 │
/// └             ┘
#[derive(Clone, Copy, PartialEq)]
pub struct Transform([Scalar; 6]);

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl fmt::Debug for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Transform {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatter = ScalarFormatter::new(fmt.precision().or(Some(6)), fmt.alternate());
        let Self([m00, m01, m02, m10, m11, m12]) = self;
        fmt.write_str("matrix(")?;
        for val in [m00, m10, m01, m11, m02] {
            fmt.write_str(formatter.format_str(*val))?;
            fmt.write_str(" ")?;
        }
        fmt.write_str(formatter.format_str(*m12))?;
        fmt.write_str(")")?;
        Ok(())
    }
}

impl Transform {
    pub fn new(
        m00: Scalar,
        m01: Scalar,
        m02: Scalar,
        m10: Scalar,
        m11: Scalar,
        m12: Scalar,
    ) -> Self {
        Self([m00, m01, m02, m10, m11, m12])
    }

    pub fn identity() -> Self {
        Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }

    /// Apply this transformation to a point
    pub fn apply(&self, point: Point) -> Point {
        let Self([m00, m01, m02, m10, m11, m12]) = self;
        let Point([x, y]) = point;
        Point([x * m00 + y * m01 + m02, x * m10 + y * m11 + m12])
    }

    /// Find the inverse transformation
    pub fn invert(&self) -> Option<Self> {
        // inv([[M, v], [0, 1]]) = [[inv(M), - inv(M) * v], [0, 1]]
        let Self([m00, m01, m02, m10, m11, m12]) = self;
        let det = m00 * m11 - m10 * m01;
        if det.abs() <= EPSILON {
            return None;
        }
        let o00 = m11 / det;
        let o01 = -m01 / det;
        let o10 = -m10 / det;
        let o11 = m00 / det;
        let o02 = -o00 * m02 - o01 * m12;
        let o12 = -o10 * m02 - o11 * m12;
        Some(Self([o00, o01, o02, o10, o11, o12]))
    }

    /// Apply translation by `[tx, ty]` before self
    pub fn pre_translate(&self, tx: Scalar, ty: Scalar) -> Self {
        self.pre_concat(Self::new_translate(tx, ty))
    }

    pub fn new_translate(tx: Scalar, ty: Scalar) -> Self {
        Self([1.0, 0.0, tx, 0.0, 1.0, ty])
    }

    /// Apply scale transformation by `[sx, sy]` before self
    pub fn pre_scale(&self, sx: Scalar, sy: Scalar) -> Self {
        self.pre_concat(Self::new_scale(sx, sy))
    }

    pub fn new_scale(sx: Scalar, sy: Scalar) -> Self {
        Self([sx, 0.0, 0.0, 0.0, sy, 0.0])
    }

    /// Apply rotation by `a` angle around the origin before self
    pub fn pre_rotate(&self, a: Scalar) -> Self {
        self.pre_concat(Self::new_rotate(a))
    }

    pub fn new_rotate(a: Scalar) -> Self {
        let (sin, cos) = a.sin_cos();
        Self([cos, -sin, 0.0, sin, cos, 0.0])
    }

    /// Apply rotation around point `p` by angle `a` before self
    pub fn pre_rotate_around(&self, a: Scalar, p: impl Into<Point>) -> Self {
        let p = p.into();
        self.pre_translate(p.x(), p.y())
            .pre_rotate(a)
            .pre_translate(-p.x(), -p.y())
    }

    /// Apply skew transformation by `[ax, ay]` before self
    pub fn pre_skew(&self, ax: Scalar, ay: Scalar) -> Self {
        self.pre_concat(Self::new_skew(ax, ay))
    }

    pub fn new_skew(ax: Scalar, ay: Scalar) -> Self {
        Self([1.0, ax.tan(), 0.0, ay.tan(), 1.0, 0.0])
    }

    /// Apply other transformation before the current one
    pub fn pre_concat(&self, other: Self) -> Self {
        *self * other
    }

    /// Apply other transformation after the current one
    pub fn post_concat(&self, other: Self) -> Self {
        other * *self
    }

    /// Create transformation which converts `src` line segment to `dst` line segment
    pub fn line_to_line(src: Line, dst: Line) -> Option<Self> {
        // Find transformation which converts (0, 0) to p0 and (0, 1) to p1
        fn unit_y_to_line(line: Line) -> Transform {
            let Line([p0, p1]) = line;
            // rotation + scale
            #[rustfmt::skip]
            let tr = Transform::new(
                p1.y() - p0.y(), p1.x() - p0.x(), p0.x(),
                p0.x() - p1.x(), p1.y() - p0.y(), p0.y(),
            );
            tr
        }
        Some(unit_y_to_line(dst) * unit_y_to_line(src).invert()?)
    }

    /// Create transformation which makes line horizontal with origin at (0, 0).
    pub fn make_horizontal(line: Line) -> Transform {
        let [p0, p1] = line.points();
        let cos_sin = match (p1 - p0).normalize() {
            None => return Transform::identity(),
            Some(cos_sin) => cos_sin,
        };
        let cos = cos_sin.x();
        let sin = cos_sin.y();
        Transform([cos, sin, 0.0, -sin, cos, 0.0]).pre_translate(-p0.x(), -p0.y())
    }

    /// Create transformation that is required to fit `src` box into `dst`.
    pub fn fit_bbox(src: BBox, dst: BBox, align: Align) -> Transform {
        let scale = (dst.height() / src.height()).min(dst.width() / src.width());
        let base = Transform::new_translate(dst.x(), dst.y())
            .pre_scale(scale, scale)
            .pre_translate(-src.x(), -src.y());
        let align = match align {
            Align::Min => Transform::identity(),
            Align::Mid => Transform::new_translate(
                (dst.width() - src.width() * scale) / 2.0,
                (dst.height() - src.height() * scale) / 2.0,
            ),
            Align::Max => Transform::new_translate(
                dst.width() - src.width() * scale,
                dst.height() - src.height() * scale,
            ),
        };
        align * base
    }

    /// Create transformation needed to fit source bounding box to provided size image
    pub fn fit_size(src: BBox, size: Size, align: Align) -> (Size, Transform) {
        let src = {
            let min = Point::new(src.min().x().floor(), src.min().y().floor());
            let max = Point::new(src.max().x().ceil(), src.max().y().ceil());
            BBox::new(min, max)
        };
        let (height, width) = match (size.height, size.width) {
            (0, 0) => (src.height(), src.width()),
            (height, 0) => {
                let height = height as Scalar;
                (height, (src.width() * height / src.height()).ceil())
            }
            (0, width) => {
                let width = width as Scalar;
                ((src.height() * width / src.width()).ceil(), width)
            }
            (height, width) => (height as Scalar, width as Scalar),
        };
        let dst = BBox::new((0.0, 0.0), (width, height));
        (
            Size {
                height: height as usize,
                width: width as usize,
            },
            Transform::fit_bbox(src, dst, align),
        )
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    /// Multiply matrices representing transformations
    fn mul(self, other: Transform) -> Self::Output {
        let Self([s00, s01, s02, s10, s11, s12]) = self;
        let Self([o00, o01, o02, o10, o11, o12]) = other;

        // s00, s01, s02 | o00, o01, o02
        // s10, s11, s12 | o10, o11, o12
        // 0  , 0  , 1   | 0  , 0  , 1
        Self([
            s00 * o00 + s01 * o10,
            s00 * o01 + s01 * o11,
            s00 * o02 + s01 * o12 + s02,
            s10 * o00 + s11 * o10,
            s10 * o01 + s11 * o11,
            s10 * o02 + s11 * o12 + s12,
        ])
    }
}

/// Bounding box with sides directed along the axes
#[derive(Clone, Copy, PartialEq, Hash)]
pub struct BBox {
    /// Point with minimal x and y values
    min: Point,
    /// Point with maximum x and y values
    max: Point,
}

impl BBox {
    /// Construct bounding box which includes points `p0` and `p1`
    pub fn new(p0: impl Into<Point>, p1: impl Into<Point>) -> Self {
        let Point([x0, y0]) = p0.into();
        let Point([x1, y1]) = p1.into();
        let (x0, x1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let (y0, y1) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
        Self {
            min: Point([x0, y0]),
            max: Point([x1, y1]),
        }
    }

    /// Point with minimum values of x and y coordinates
    #[inline]
    pub fn min(&self) -> Point {
        self.min
    }

    /// Point with minimum values of x and y coordinates
    #[inline]
    pub fn max(&self) -> Point {
        self.max
    }

    /// `x` coordinate of the point with the minimal value
    #[inline]
    pub fn x(&self) -> Scalar {
        self.min.x()
    }

    /// `y` coordinate of the point with the minimal value
    #[inline]
    pub fn y(&self) -> Scalar {
        self.min.y()
    }

    /// Width of the bounding box
    #[inline]
    pub fn width(&self) -> Scalar {
        self.max.x() - self.min.x()
    }

    /// Hight of the bounding box
    #[inline]
    pub fn height(&self) -> Scalar {
        self.max.y() - self.min.y()
    }

    /// Diagonal line from `min` to `max` of the bounding box
    pub fn diag(&self) -> Line {
        Line::new(self.min, self.max)
    }

    /// Determine if the point is inside of the bounding box
    pub fn contains(&self, point: Point) -> bool {
        let Point([x, y]) = point;
        self.min.x() <= x && x <= self.max.x() && self.min.y() <= y && y <= self.max.y()
    }

    /// Extend bounding box so it would contains provided point
    pub fn extend(&self, point: Point) -> Self {
        let Point([x, y]) = point;
        let Point([x0, y0]) = self.min;
        let Point([x1, y1]) = self.max;
        let (x0, x1) = if x < x0 {
            (x, x1)
        } else if x > x1 {
            (x0, x)
        } else {
            (x0, x1)
        };
        let (y0, y1) = if y < y0 {
            (y, y1)
        } else if y > y1 {
            (y0, y)
        } else {
            (y0, y1)
        };
        Self {
            min: Point([x0, y0]),
            max: Point([x1, y1]),
        }
    }

    /// Create union bounding box of two bounding boxes
    pub fn union(&self, other: BBox) -> Self {
        self.extend(other.min).extend(other.max)
    }

    pub fn union_opt(&self, other: Option<BBox>) -> Self {
        match other {
            Some(other) => self.union(other),
            None => *self,
        }
    }

    /// Find bounding box of the intersection of two bounding boxes
    pub fn intersect(&self, other: BBox) -> Option<BBox> {
        let (x_min, x_max) =
            range_intersect(self.min.x(), self.max.x(), other.min.x(), other.max.x())?;
        let (y_min, y_max) =
            range_intersect(self.min.y(), self.max.y(), other.min.y(), other.max.y())?;
        Some(BBox::new(
            Point::new(x_min, y_min),
            Point::new(x_max, y_max),
        ))
    }

    /// Transform that makes bounding box a unit-sized square
    ///
    /// This is used by clip|mask|gradient units
    pub fn unit_transform(&self) -> Transform {
        Transform::new_translate(self.x(), self.y()).pre_scale(self.width(), self.height())
    }

    /// Compute new bounding box such that it will include original bounding box after transformation
    pub fn transform(&self, tr: Transform) -> BBox {
        let p00 = tr.apply(self.min);
        let p01 = tr.apply(Point::new(self.min.x(), self.max.y()));
        let p10 = tr.apply(Point::new(self.max.x(), self.min.y()));
        let p11 = tr.apply(self.max);
        BBox::new(p00, p11).extend(p10).extend(p01)
    }
}

/// Find intersection of two ranges
fn range_intersect(
    r0_min: Scalar,
    r0_max: Scalar,
    r1_min: Scalar,
    r1_max: Scalar,
) -> Option<(Scalar, Scalar)> {
    if r0_min > r1_max || r1_min > r0_max {
        None
    } else {
        Some((r0_min.max(r1_min), r0_max.min(r1_max)))
    }
}

impl FromStr for BBox {
    type Err = SvgParserError;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        let mut values = string.split([' ', ',']).map(|s| s.trim().parse().ok());
        let minx: Scalar = values.next().flatten().ok_or(SvgParserError::InvalidBBox)?;
        let miny: Scalar = values.next().flatten().ok_or(SvgParserError::InvalidBBox)?;
        let width: Scalar = values.next().flatten().ok_or(SvgParserError::InvalidBBox)?;
        let height: Scalar = values.next().flatten().ok_or(SvgParserError::InvalidBBox)?;
        Ok(BBox::new((minx, miny), (minx + width, miny + height)))
    }
}

impl fmt::Display for BBox {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatter = ScalarFormatter::new_fmt(fmt);
        fmt.write_str(formatter.format_str(self.x()))?;
        fmt.write_str(" ")?;
        fmt.write_str(formatter.format_str(self.y()))?;
        fmt.write_str(" ")?;
        fmt.write_str(formatter.format_str(self.width()))?;
        fmt.write_str(" ")?;
        fmt.write_str(formatter.format_str(self.height()))
    }
}

impl fmt::Debug for BBox {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatter = ScalarFormatter::new_fmt(fmt);
        fmt.debug_struct("BBox")
            .field("x", &formatter.format_str(self.x()))
            .field("y", &formatter.format_str(self.y()))
            .field("w", &formatter.format_str(self.width()))
            .field("h", &formatter.format_str(self.height()))
            .finish()
    }
}

#[cfg(feature = "serde")]
impl Serialize for BBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.x(), self.y(), self.width(), self.height()).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for BBox {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (minx, miny, width, height): (Scalar, Scalar, Scalar, Scalar) =
            Deserialize::deserialize(deserializer)?;
        Ok(BBox::new((minx, miny), (minx + width, miny + height)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Curve, assert_approx_eq};
    type Error = Box<dyn std::error::Error>;

    #[test]
    fn test_trasform() -> Result<(), Error> {
        let tr = Transform::identity()
            .pre_translate(1.0, 2.0)
            .pre_rotate(PI / 3.0)
            .pre_skew(2.0, 3.0)
            .pre_scale(3.0, 2.0);
        let inv = tr.invert().unwrap();
        let p0 = Point::new(1.0, 1.0);

        let p1 = tr.apply(p0);
        assert_approx_eq!(p1.x(), -1.04674389, 1e-6);
        assert_approx_eq!(p1.y(), 1.59965634, 1e-6);

        let p2 = inv.apply(p1);
        assert_approx_eq!(p2.x(), 1.0, 1e-6);
        assert_approx_eq!(p2.y(), 1.0, 1e-6);

        let l0 = Line::new((1.0, 0.0), (-3.0, 3.0));
        let l1 = l0.transform(Transform::make_horizontal(l0));
        assert_eq!(l1.start(), Point::new(0.0, 0.0));
        assert_approx_eq!(l1.end().x(), 5.0);
        assert_approx_eq!(l1.end().y(), 0.0, 1e-6);

        let s0 = Line::new((2.0, 1.0), (1.0, 4.0));
        // unit vector perpendicular to s0
        let s1 = Line::new(
            s0.start(),
            s0.start() + s0.direction().normal().normalize().unwrap(),
        );
        let d0 = Line::new((3.0, 1.0), (4.0, 2.0));
        let tr = Transform::line_to_line(s0, d0).unwrap();
        let o0 = s0.transform(tr);
        let o1 = s1.transform(tr);
        assert_approx_eq!((o0.start() - d0.start()).length(), 0.0);
        assert_approx_eq!((o0.end() - d0.end()).length(), 0.0);
        // no skew introduced
        assert_approx_eq!(o0.direction().dot(o1.direction()), 0.0);
        // uniform scale
        assert_approx_eq!(o1.length(), d0.length() / s0.length());

        Ok(())
    }

    #[test]
    fn test_transform_fit() -> Result<(), Error> {
        let s0 = BBox::new(Point::new(1.0, 1.0), Point::new(2.0, 2.0));
        let s1 = BBox::new(Point::new(1.0, 1.0), Point::new(1.5, 2.0));
        let s2 = BBox::new(Point::new(1.0, 1.0), Point::new(2.0, 1.5));
        let d = BBox::new(Point::new(3.0, 5.0), Point::new(13.0, 15.0));

        let tr0 = Transform::fit_bbox(s0, d, Align::Mid);
        assert!(tr0.apply(s0.min).is_close_to(d.min));
        assert!(tr0.apply(s0.max).is_close_to(d.max));

        let tr1 = Transform::fit_bbox(s1, d, Align::Min);
        assert!(tr1.apply(s1.min).is_close_to(d.min));
        assert!(tr1.apply(s1.max).is_close_to(Point::new(8.0, 15.0)));

        let tr2 = Transform::fit_bbox(s2, d, Align::Max);
        assert!(tr2.apply(s2.max).is_close_to(d.max));
        assert!(tr2.apply(s2.min).is_close_to(Point::new(3.0, 10.0)));

        let tr3 = Transform::fit_bbox(s1, d, Align::Mid);
        assert!(
            tr3.apply((s1.min + s1.max) / 2.0)
                .is_close_to((d.min + d.max) / 2.0)
        );
        assert!(tr3.apply(s1.min).is_close_to(Point::new(5.5, 5.0)));
        assert!(tr3.apply(s1.max).is_close_to(Point::new(10.5, 15.0)));

        let tr4 = Transform::fit_bbox(s2, d, Align::Mid);
        assert!(
            tr4.apply((s2.min + s2.max) / 2.0)
                .is_close_to((d.min + d.max) / 2.0)
        );
        assert!(tr4.apply(s2.min).is_close_to(Point::new(3.0, 7.5)));
        assert!(tr4.apply(s2.max).is_close_to(Point::new(13.0, 12.5)));

        Ok(())
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_bbox_serde() -> Result<(), Error> {
        let expected = BBox::new((1.0, 2.0), (4.0, 6.0));
        let result: BBox = serde_json::from_str(&serde_json::to_string(&expected)?)?;
        assert_eq!(result, expected);
        let result: BBox = serde_json::from_str("[1, 2, 3, 4]")?;
        assert_eq!(result, expected);

        assert_eq!(expected, expected.to_string().parse()?);
        Ok(())
    }

    #[test]
    fn test_scalar_format() -> Result<(), Error> {
        let value: Scalar = 0.1234567;
        assert_eq!(format!("{}", ScalarFormat(value)), "0.1235".to_owned());
        assert_eq!(format!("{:#}", ScalarFormat(value)), "0.1235".to_owned());
        assert_eq!(format!("{:.3}", ScalarFormat(value)), "0.123".to_owned());

        let value: Scalar = 12.3001;
        assert_eq!(format!("{}", ScalarFormat(value)), "12.30".to_owned());
        assert_eq!(format!("{:#}", ScalarFormat(value)), "12.3".to_owned());
        assert_eq!(format!("{:.3}", ScalarFormat(value)), "12.3".to_owned());

        let value: Scalar = 12300.0;
        assert_eq!(format!("{}", ScalarFormat(value)), "12300".to_owned());
        assert_eq!(format!("{:#}", ScalarFormat(value)), "12300".to_owned());
        assert_eq!(format!("{:.3}", ScalarFormat(value)), "12300".to_owned());

        let value: Scalar = 0.0;
        assert_eq!(format!("{}", ScalarFormat(value)), "0".to_owned());
        assert_eq!(format!("{:#}", ScalarFormat(value)), "0".to_owned());
        assert_eq!(format!("{:.3}", ScalarFormat(value)), "0".to_owned());

        Ok(())
    }
}
