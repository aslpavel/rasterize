use crate::{clamp, Line};
use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
};

pub type Scalar = f64;
pub const EPSILON: f64 = std::f64::EPSILON;
pub const EPSILON_SQRT: f64 = 1.490_116_119_384_765_6e-8;
pub const PI: f64 = std::f64::consts::PI;

/// Format floats in a compact way suitable for SVG path
pub fn scalar_fmt(f: &mut fmt::Formatter<'_>, value: Scalar) -> fmt::Result {
    let value_abs = value.abs();
    if value_abs.fract() < EPSILON {
        write!(f, "{}", value.trunc() as i64)
    } else if value_abs > 9999.0 || value_abs <= 0.0001 {
        write!(f, "{:.3e}", value)
    } else {
        let ten: Scalar = 10.0;
        let round = ten.powi(6 - (value_abs.trunc() + 1.0).log10().ceil() as i32);
        write!(f, "{}", (value * round).round() / round)
    }
}

/// Value representing a 2D point or vector.
#[derive(Clone, Copy, PartialEq)]
pub struct Point(pub [Scalar; 2]);

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Point([x, y]) = self;
        scalar_fmt(f, *x)?;
        write!(f, ",")?;
        scalar_fmt(f, *y)?;
        Ok(())
    }
}

impl Point {
    #[inline]
    pub fn new(x: Scalar, y: Scalar) -> Self {
        Self([x, y])
    }

    /// Get `x` component of the point
    #[inline]
    pub fn x(&self) -> Scalar {
        self.0[0]
    }

    /// Get `y` compenent of the point
    #[inline]
    pub fn y(self) -> Scalar {
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

    /// Determine if self is close to the other within the marging of error (EPSILON)
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Align {
    /// Alling by minimal value
    Min,
    /// Aling by center value
    Mid,
    /// Align by maximum value
    Max,
}

/// 2D affine transformation
///
/// Stored as an array [m00, m01, m02, m10, m11, m12] but semantically corresponds to
/// a matrix:
/// ┌             ┐
/// │ m00 m01 m02 │
/// │ m11 m11 m12 │
/// │   0   0   1 │
/// └             ┘
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform([Scalar; 6]);

impl Default for Transform {
    fn default() -> Self {
        Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }
}

impl Transform {
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
    pub fn translate(&self, tx: Scalar, ty: Scalar) -> Self {
        self.matmul(Self([1.0, 0.0, tx, 0.0, 1.0, ty]))
    }

    /// Apply scale transformatoin by `[sx, sy]` before self
    pub fn scale(&self, sx: Scalar, sy: Scalar) -> Self {
        self.matmul(Self([sx, 0.0, 0.0, 0.0, sy, 0.0]))
    }

    /// Apply rotation by `a` angle around the origin before self
    pub fn rotate(&self, a: Scalar) -> Self {
        let (sin, cos) = a.sin_cos();
        self.matmul(Self([cos, -sin, 0.0, sin, cos, 0.0]))
    }

    /// Apply rotation around point `p` by angle `a` before self
    pub fn rotate_around(&self, a: Scalar, p: impl Into<Point>) -> Self {
        let p = p.into();
        self.translate(p.x(), p.y())
            .rotate(a)
            .translate(-p.x(), -p.y())
    }

    /// Apply scew transformation by `[ax, ay]` before self
    pub fn skew(&self, ax: Scalar, ay: Scalar) -> Self {
        self.matmul(Self([1.0, ax.tan(), 0.0, ay.tan(), 1.0, 0.0]))
    }

    /// Multiply transformations in matrix form
    pub fn matmul(&self, other: Transform) -> Self {
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

    /// Find transformation which makes line horizontal with origin at (0, 0).
    pub fn make_horizontal(line: Line) -> Transform {
        let [p0, p1] = line.points();
        let cos_sin = match (p1 - p0).normalize() {
            None => return Transform::default(),
            Some(cos_sin) => cos_sin,
        };
        let cos = cos_sin.x();
        let sin = cos_sin.y();
        Transform::default()
            .matmul(Self([cos, sin, 0.0, -sin, cos, 0.0]))
            .translate(-p0.x(), -p0.y())
    }

    /// Find transformation that is requred to fit `src` box into `dst`.
    pub fn fit(src: BBox, dst: BBox, align: Align) -> Transform {
        let scale = (dst.height() / src.height()).min(dst.width() / src.width());
        let base = Transform::default()
            .translate(dst.x(), dst.y())
            .scale(scale, scale)
            .translate(-src.x(), -src.y());
        let align = match align {
            Align::Min => Transform::default(),
            Align::Mid => Transform::default().translate(
                (dst.width() - src.width() * scale) / 2.0,
                (dst.height() - src.height() * scale) / 2.0,
            ),
            Align::Max => Transform::default().translate(
                dst.width() - src.width() * scale,
                dst.height() - src.height() * scale,
            ),
        };
        align * base
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    fn mul(self, other: Transform) -> Self::Output {
        self.matmul(other)
    }
}

/// Bounding box with sides directed along the axes
#[derive(Clone, Copy)]
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

    /// Point with minmum values of x and y coordianetes
    #[inline]
    pub fn min(&self) -> Point {
        self.min
    }

    /// Point with minmum values of x and y coordianetes
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

    /// Create bounding box the spans both bbox-es
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

impl fmt::Debug for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BBox x=")?;
        scalar_fmt(f, self.x())?;
        write!(f, ", y=")?;
        scalar_fmt(f, self.y())?;
        write!(f, ", w=")?;
        scalar_fmt(f, self.width())?;
        write!(f, ", h=")?;
        scalar_fmt(f, self.height())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq, Curve};

    #[test]
    fn test_trasform() {
        let tr = Transform::default()
            .translate(1.0, 2.0)
            .rotate(PI / 3.0)
            .skew(2.0, 3.0)
            .scale(3.0, 2.0);
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
    }

    #[test]
    fn test_transform_fit() {
        let s0 = BBox::new(Point::new(1.0, 1.0), Point::new(2.0, 2.0));
        let s1 = BBox::new(Point::new(1.0, 1.0), Point::new(1.5, 2.0));
        let s2 = BBox::new(Point::new(1.0, 1.0), Point::new(2.0, 1.5));
        let d = BBox::new(Point::new(3.0, 5.0), Point::new(13.0, 15.0));

        let tr0 = Transform::fit(s0, d, Align::Mid);
        assert!(tr0.apply(s0.min).is_close_to(d.min));
        assert!(tr0.apply(s0.max).is_close_to(d.max));

        let tr1 = Transform::fit(s1, d, Align::Min);
        assert!(tr1.apply(s1.min).is_close_to(d.min));
        assert!(tr1.apply(s1.max).is_close_to(Point::new(8.0, 15.0)));

        let tr2 = Transform::fit(s2, d, Align::Max);
        assert!(tr2.apply(s2.max).is_close_to(d.max));
        assert!(tr2.apply(s2.min).is_close_to(Point::new(3.0, 10.0)));

        let tr3 = Transform::fit(s1, d, Align::Mid);
        assert!(tr3
            .apply((s1.min + s1.max) / 2.0)
            .is_close_to((d.min + d.max) / 2.0));
        assert!(tr3.apply(s1.min).is_close_to(Point::new(5.5, 5.0)));
        assert!(tr3.apply(s1.max).is_close_to(Point::new(10.5, 15.0)));

        let tr4 = Transform::fit(s2, d, Align::Mid);
        assert!(tr4
            .apply((s2.min + s2.max) / 2.0)
            .is_close_to((d.min + d.max) / 2.0));
        assert!(tr4.apply(s2.min).is_close_to(Point::new(3.0, 7.5)));
        assert!(tr4.apply(s2.max).is_close_to(Point::new(13.0, 12.5)));
    }
}
