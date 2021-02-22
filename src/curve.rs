//! All the things you need to handle bezier curves

use crate::{
    cubic_solve, quadratic_solve, ArrayIter, BBox, EllipArc, Error, LineCap, LineJoin, M3x3, M4x4,
    Path, Point, Scalar, StrokeStyle, Transform, EPSILON,
};
use std::{fmt, str::FromStr};

pub type CurveRoots = ArrayIter<[Option<Scalar>; 3]>;
pub type CurveExtremities = ArrayIter<[Option<Scalar>; 6]>;

/// Set of operations common to all bezier curves.
pub trait Curve: Sized + Into<Segment> {
    /// Convert curve to an iterator over line segments with desired flatness
    fn flatten(&self, tr: Transform, flatness: Scalar) -> CurveFlattenIter {
        CurveFlattenIter::new(self.transform(tr), flatness)
    }

    /// Correspond to maximum diviation of the curve from the straight line
    /// `f = max |curve(t) - line(curve_start, curve_end)(t)|`. This function
    /// actually returns `16.0 * f^2` to avoid unneeded division and square root.
    fn flatness(&self) -> Scalar;

    /// Apply affine transformation to the curve
    fn transform(&self, tr: Transform) -> Self;

    /// Point at which curve starts
    fn start(&self) -> Point;

    /// Point at which curve ends
    fn end(&self) -> Point;

    /// Evaluate curve at parameter value `t` in (0.0..=1.0)
    fn at(&self, t: Scalar) -> Point;

    /// Optimized version of `Curve::split_at(0.5)`
    fn split(&self) -> (Self, Self) {
        self.split_at(0.5)
    }

    /// Split the curve at prameter value `t`
    fn split_at(&self, t: Scalar) -> (Self, Self);

    /// Create subcurve specified starting at parameter value `a` and ending at value `b`
    fn cut(&self, a: Scalar, b: Scalar) -> Self;

    /// Extend provided `init` bounding box with the bounding box of the curve
    fn bbox(&self, init: Option<BBox>) -> BBox;

    /// Offset the curve by distance `dist`, result is inserted into `out` container
    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>);

    /// Derivative with respect to t, `deriv(t) = [curve'(t)_x, curve'(t)_y]`
    fn deriv(&self) -> Segment;

    /// Identical curve but directed from end to start, instead of start to end.
    fn reverse(&self) -> Self;

    /// Find roots of the equation `curve(t)_y = 0`. Values of the parameter at which curve
    /// crosses y axis.
    fn roots(&self) -> CurveRoots;

    /// Find all extermities of the curve `curve'(t)_x = 0 || curve'(t)_y = 0`
    fn extremities(&self) -> CurveExtremities;
}

pub struct CurveFlattenIter {
    flatness: Scalar,
    stack: Vec<Segment>,
}

impl CurveFlattenIter {
    pub fn new(segment: impl Into<Segment>, flatness: Scalar) -> Self {
        Self {
            flatness: 16.0 * flatness * flatness,
            stack: vec![segment.into()],
        }
    }
}

impl Iterator for CurveFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stack.pop() {
                None => {
                    return None;
                }
                Some(segment) => {
                    if segment.flatness() < self.flatness {
                        return Some(Line([segment.start(), segment.end()]));
                    }
                    let (s0, s1) = segment.split();
                    self.stack.push(s1);
                    self.stack.push(s0);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Line
// -----------------------------------------------------------------------------

/// Line segment curve
#[derive(Clone, Copy, PartialEq)]
pub struct Line(pub [Point; 2]);

impl fmt::Debug for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Line([p0, p1]) = self;
        write!(f, "Line {:?} {:?}", p0, p1)
    }
}

impl Line {
    pub fn new(p0: impl Into<Point>, p1: impl Into<Point>) -> Self {
        Self([p0.into(), p1.into()])
    }

    /// Length of the line
    pub fn length(&self) -> Scalar {
        let Self([p0, p1]) = self;
        p0.dist(*p1)
    }

    /// Start and end points of the line
    pub fn points(&self) -> [Point; 2] {
        self.0
    }

    pub fn ends(&self) -> (Line, Line) {
        (*self, *self)
    }

    /// Find intersection of two lines
    ///
    /// Returns pair of `t` parameters for this line and the other line.
    /// Found by solving `self.at(t0) == other.at(t1)`. Actual intersection of
    /// line segments can be found by making sure that `0.0 <= t0 <= 1.0 && 0.0 <= t1 <= 1.0`
    pub fn intersect(&self, other: Line) -> Option<(Scalar, Scalar)> {
        let Line([Point([x1, y1]), Point([x2, y2])]) = *self;
        let Line([Point([x3, y3]), Point([x4, y4])]) = other;
        let det = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3);
        if det.abs() < EPSILON {
            return None;
        }
        let t0 = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / det;
        let t1 = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / det;
        Some((t0, t1))
    }

    /// Find intersection point between two line segments
    pub fn intersect_point(&self, other: Line) -> Option<Point> {
        let (t0, t1) = self.intersect(other)?;
        if (0.0..=1.0).contains(&t0) && (0.0..=1.0).contains(&t1) {
            Some(self.at(t0))
        } else {
            None
        }
    }

    /// Direction vector associated with the line segment
    pub fn direction(&self) -> Point {
        self.end() - self.start()
    }
}

impl Curve for Line {
    fn flatness(&self) -> Scalar {
        0.0
    }

    fn transform(&self, tr: Transform) -> Self {
        let Line([p0, p1]) = self;
        Self([tr.apply(*p0), tr.apply(*p1)])
    }

    fn start(&self) -> Point {
        self.0[0]
    }

    fn end(&self) -> Point {
        self.0[1]
    }

    fn at(&self, t: Scalar) -> Point {
        let Self([p0, p1]) = self;
        (1.0 - t) * p0 + t * p1
    }

    fn deriv(&self) -> Segment {
        let deriv = self.end() - self.start();
        Line::new(deriv, deriv).into()
    }

    fn split_at(&self, t: Scalar) -> (Self, Self) {
        let Self([p0, p1]) = self;
        let mid = self.at(t);
        (Self([*p0, mid]), Self([mid, *p1]))
    }

    fn cut(&self, a: Scalar, b: Scalar) -> Self {
        Self([self.at(a), self.at(b)])
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        let Self([p0, p1]) = *self;
        BBox::new(p0, p1).union_opt(init)
    }

    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>) {
        out.extend(line_offset(*self, dist).map(Segment::from));
    }

    fn reverse(&self) -> Self {
        let Self([p0, p1]) = *self;
        Self([p1, p0])
    }

    fn roots(&self) -> CurveRoots {
        let mut result = CurveRoots::new();
        let Self([Point([_, y0]), Point([_, y1])]) = self;
        if (y0 - y1).abs() > EPSILON {
            let t = y0 / (y0 - y1);
            if (0.0..=1.0).contains(&t) {
                result.push(t);
            }
        }
        result
    }

    fn extremities(&self) -> CurveExtremities {
        CurveExtremities::new()
    }
}

impl FromStr for Line {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let segment = Segment::from_str(text)?;
        segment.to_line().ok_or_else(|| Error::ConvertionError {
            reason: "first element of the path is not a line".to_string(),
        })
    }
}

// -----------------------------------------------------------------------------
// Quadratic bezier curve
// -----------------------------------------------------------------------------

// Matrix form for quadratic bezier curve
#[rustfmt::skip]
const Q: M3x3 = M3x3([
    1.0,  0.0, 0.0,
   -2.0,  2.0, 0.0,
    1.0, -2.0, 1.0,
]);

// Inverted matrix form for quadratic bezier curve
#[rustfmt::skip]
const QI: M3x3 = M3x3([
    1.0, 0.0, 0.0,
    1.0, 0.5, 0.0,
    1.0, 1.0, 1.0,
]);

/// Quadratic bezier curve
///
/// Polynimial form:
/// `(1 - t) ^ 2 * p0 + 2 * (1 - t) * t * p1 + t ^ 2 * p2`
/// Matrix from:
///             ┌          ┐ ┌    ┐
/// ┌         ┐ │  1  0  0 │ │ p0 │
/// │ 1 t t^2 │ │ -2  2  0 │ │ p1 │
/// └         ┘ │  1 -2  1 │ │ p2 │
///             └          ┘ └    ┘
#[derive(Clone, Copy, PartialEq)]
pub struct Quad(pub [Point; 3]);

impl fmt::Debug for Quad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Quad([p0, p1, p2]) = self;
        write!(f, "Quad {:?} {:?} {:?}", p0, p1, p2)
    }
}

impl Quad {
    pub fn new(p0: impl Into<Point>, p1: impl Into<Point>, p2: impl Into<Point>) -> Self {
        Self([p0.into(), p1.into(), p2.into()])
    }

    pub fn points(&self) -> [Point; 3] {
        self.0
    }

    pub fn ends(&self) -> (Line, Line) {
        let Self([p0, p1, p2]) = *self;
        let start = Line::new(p0, p1);
        let end = Line::new(p1, p2);
        if p0.is_close_to(p1) {
            (end, end)
        } else if p1.is_close_to(p2) {
            (start, start)
        } else {
            (start, end)
        }
    }

    /// Find smooth point used by SVG parser
    pub fn smooth(&self) -> Point {
        let Quad([_p0, p1, p2]) = self;
        2.0 * p2 - *p1
    }
}

impl Curve for Quad {
    /// Flattness criteria for the cubic curve
    ///
    /// It is equal to `f = max d(t) where d(t) = |q(t) - l(t)|, l(t) = (1 - t) * p0 + t * p2`
    /// for q(t) bezier2 curve with p{0..2} control points, in other words maximum distance
    /// from parametric line to bezier2 curve for the same parameter t.
    ///
    /// Line can be represented as bezier2 curve, if `p1 = (p0 + p2) / 2.0`.
    /// Grouping polynomial coofficients:
    ///     q(t) = t^2 p2 + 2 (1 - t) t p1 + (1 - t)^2 p0
    ///     l(t) = t^2 p2 + (1 - t) t (p0 + p2) + (1 - t)^2 p0
    ///     d(t) = |q(t) - l(t)| = (1 - t) t |2 * p1 - p0 - p2|
    ///     f    = 1 / 4 * | 2 p1 - p0 - p2 |
    ///     f^2  = 1/16 |2 * p1 - p0 - p2|^2
    ///
    fn flatness(&self) -> Scalar {
        let Self([p0, p1, p2]) = *self;
        let Point([x, y]) = 2.0 * p1 - p0 - p2;
        x * x + y * y
    }

    fn transform(&self, tr: Transform) -> Self {
        let Quad([p0, p1, p2]) = self;
        Self([tr.apply(*p0), tr.apply(*p1), tr.apply(*p2)])
    }

    fn start(&self) -> Point {
        self.0[0]
    }

    fn end(&self) -> Point {
        self.0[2]
    }

    fn at(&self, t: Scalar) -> Point {
        // at(t) =
        //   (1 - t) ^ 2 * p0 +
        //   2 * (1 - t) * t * p1 +
        //   t ^ 2 * p2
        let Self([p0, p1, p2]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        t_2 * p0 + 2.0 * t1 * t_1 * p1 + t2 * p2
    }

    fn deriv(&self) -> Segment {
        let Self([p0, p1, p2]) = *self;
        Line::new(2.0 * (p1 - p0), 2.0 * (p2 - p1)).into()
    }

    /// Optimized version of `split_at(0.5)`
    fn split(&self) -> (Self, Self) {
        let Self([p0, p1, p2]) = *self;
        let mid = 0.25 * (p0 + 2.0 * p1 + p2);
        (
            Self([p0, 0.5 * (p0 + p1), mid]),
            Self([mid, 0.5 * (p1 + p2), p2]),
        )
    }

    fn split_at(&self, t: Scalar) -> (Self, Self) {
        // https://pomax.github.io/bezierinfo/#matrixsplit
        let Self([p0, p1, p2]) = *self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        let mid = t_2 * p0 + 2.0 * t1 * t_1 * p1 + t2 * p2;
        (
            Self([p0, t_1 * p0 + t * p1, mid]),
            Self([mid, t_1 * p1 + t * p2, p2]),
        )
    }

    fn cut(&self, a: Scalar, b: Scalar) -> Self {
        // Given curve as Q(t) = [1 t t^2] M Q
        // we can change parameter t -> a + (b - a) * t which will produced desired curve
        // it is possible to decompose it as
        //             ┌                         ┐
        // ┌         ┐ │  1  a       a^2         │
        // │ 1 t t^2 │ │  0  (b - a) 2*a*(b - a) │ = [1 t t^2] T
        // └         ┘ │  0  0       (b - a)^2   │
        //             └                         ┘
        // we can convert it back to desired curve by Q[a, b](t) = [1 t t^2] Q (QI T Q) P
        let Self([p0, p1, p2]) = self;
        let ba = b - a;
        #[rustfmt::skip]
        let t = M3x3([
            1.0, a  , a * a       ,
            0.0, ba , 2.0 * a * ba,
            0.0, 0.0, ba * ba     ,
        ]);
        #[rustfmt::skip]
        let M3x3([
            m00, m01, m02,
            m10, m11, m12,
            m20, m21, m22,
        ]) = QI * t * Q;
        let q0 = m00 * p0 + m01 * p1 + m02 * p2;
        let q1 = m10 * p0 + m11 * p1 + m12 * p2;
        let q2 = m20 * p0 + m21 * p1 + m22 * p2;
        Self([q0, q1, q2])
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        let Self([p0, p1, p2]) = self;
        let bbox = BBox::new(*p0, *p2).union_opt(init);
        if bbox.contains(*p1) {
            return bbox;
        }
        self.extremities()
            .fold(bbox, |bbox, t| bbox.extend(self.at(t)))
    }

    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>) {
        quad_offset_rec(*self, dist, out, 0)
    }

    fn reverse(&self) -> Self {
        let Self([p0, p1, p2]) = *self;
        Self([p2, p1, p0])
    }

    fn roots(&self) -> CurveRoots {
        let mut result = CurveRoots::new();
        // curve(t)_y = 0
        let Self([Point([_, y0]), Point([_, y1]), Point([_, y2])]) = *self;
        let a = y0 - 2.0 * y1 + y2;
        let b = -2.0 * y0 + 2.0 * y1;
        let c = y0;
        result.extend(quadratic_solve(a, b, c).filter(|t| (0.0..=1.0).contains(t)));
        result
    }

    fn extremities(&self) -> CurveExtremities {
        let mut result = CurveExtremities::new();
        let Self([p0, p1, p2]) = self;
        let Point([a0, a1]) = *p2 - 2.0 * p1 + *p0;
        let Point([b0, b1]) = *p1 - *p0;
        // curve'(t)_x = 0
        if a0.abs() > EPSILON {
            let t0 = -b0 / a0;
            if (0.0..=1.0).contains(&t0) {
                result.push(t0)
            }
        }
        // curve'(t)_y = 0
        if a1.abs() > EPSILON {
            let t1 = -b1 / a1;
            if (0.0..=1.0).contains(&t1) {
                result.push(t1)
            }
        }
        result
    }
}

impl FromStr for Quad {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let segment = Segment::from_str(text)?;
        segment.to_quad().ok_or_else(|| Error::ConvertionError {
            reason: "first element of the path is not a quad".to_string(),
        })
    }
}

// -----------------------------------------------------------------------------
// Cubic bezier curve
// -----------------------------------------------------------------------------

/// Matrix form for cubic bezier curve
#[rustfmt::skip]
const C: M4x4 = M4x4([
    1.0,  0.0,  0.0, 0.0,
   -3.0,  3.0,  0.0, 0.0,
    3.0, -6.0,  3.0, 0.0,
   -1.0,  3.0, -3.0, 1.0,
]);

/// Inverted matrix form for cubic bezier curve
#[rustfmt::skip]
const CI: M4x4 = M4x4([
    1.0, 0.0      , 0.0      , 0.0,
    1.0, 1.0 / 3.0, 0.0      , 0.0,
    1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0,
    1.0, 1.0      , 1.0      , 1.0,
]);

/// Cubic bezier curve
///
/// Polynimial form:
/// `(1 - t) ^ 3 * p0 + 3 * (1 - t) ^ 2 * t * p1 + 3 * (1 - t) * t ^ 2 * p2 + t ^ 3 * p3`
/// Matrix from:
///                 ┌             ┐ ┌    ┐
/// ┌             ┐ │  1  0  0  0 │ │ p0 │
/// │ 1 t t^2 t^3 │ │ -3  3  0  0 │ │ p1 │
/// └             ┘ │  3 -6  3  0 │ │ p2 │
///                 │ -1  3 -3  1 │ │ p3 │
///                 └             ┘ └    ┘
#[derive(Clone, Copy, PartialEq)]
pub struct Cubic(pub [Point; 4]);

impl fmt::Debug for Cubic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Cubic([p0, p1, p2, p3]) = self;
        write!(f, "Cubic {:?} {:?} {:?} {:?}", p0, p1, p2, p3)
    }
}

impl Cubic {
    pub fn new(
        p0: impl Into<Point>,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
        p3: impl Into<Point>,
    ) -> Self {
        Self([p0.into(), p1.into(), p2.into(), p3.into()])
    }

    pub fn points(&self) -> [Point; 4] {
        self.0
    }

    pub fn ends(&self) -> (Line, Line) {
        let ps = self.points();
        let mut start = 0;
        for i in 0..3 {
            if !ps[i].is_close_to(ps[i + 1]) {
                start = i;
                break;
            }
        }
        let mut end = 0;
        for i in (1..4).rev() {
            if !ps[i].is_close_to(ps[i - 1]) {
                end = i;
                break;
            }
        }
        (
            Line::new(ps[start], ps[start + 1]),
            Line::new(ps[end - 1], ps[end]),
        )
    }

    /// Find smooth point used by SVG parser
    pub fn smooth(&self) -> Point {
        let Cubic([_p0, _p1, p2, p3]) = self;
        2.0 * p3 - *p2
    }
}

impl Curve for Cubic {
    /// Flattness criteria for the cubic curve
    /// This function actually returns `16 * flatness^2`
    ///
    /// It is equal to `f = max d(t) where d(t) = |c(t) - l(t)|, l(t) = (1 - t) * c0 + t * c3`
    /// for c(t) bezier3 curve with c{0..3} control points, in other words maximum distance
    /// from parametric line to bezier3 curve for the same parameter t. It is shown in the article
    /// that:
    ///     f^2 <= 1/16 (max{u_x^2, v_x^2} + max{u_y^2, v_y^2})
    /// where:
    ///     u = 3 * b1 - 2 * b0 - b3
    ///     v = 3 * b2 - b0 - 2 * b3
    /// `f == 0` means completely flat so estimating upper bound is sufficient as spliting more
    /// than needed is not a problem for rendering.
    ///
    /// [Linear Approximation of Bezier Curve](https://hcklbrrfnn.files.wordpress.com/2012/08/bez.pdf)
    fn flatness(&self) -> Scalar {
        let Self([p0, p1, p2, p3]) = *self;
        let u = 3.0 * p1 - 2.0 * p0 - p3;
        let v = 3.0 * p2 - p0 - 2.0 * p3;
        (u.x() * u.x()).max(v.x() * v.x()) + (u.y() * u.y()).max(v.y() * v.y())
    }

    fn transform(&self, tr: Transform) -> Self {
        let Cubic([p0, p1, p2, p3]) = self;
        Self([tr.apply(*p0), tr.apply(*p1), tr.apply(*p2), tr.apply(*p3)])
    }

    fn start(&self) -> Point {
        self.0[0]
    }

    fn end(&self) -> Point {
        self.0[3]
    }

    fn at(&self, t: Scalar) -> Point {
        // at(t) =
        //   (1 - t) ^ 3 * p0 +
        //   3 * (1 - t) ^ 2 * t * p1 +
        //   3 * (1 - t) * t ^ 2 * p2 +
        //   t ^ 3 * p3
        let Self([p0, p1, p2, p3]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        let (t3, t_3) = (t2 * t1, t_2 * t_1);
        t_3 * p0 + 3.0 * t1 * t_2 * p1 + 3.0 * t2 * t_1 * p2 + t3 * p3
    }

    fn deriv(&self) -> Segment {
        let Self([p0, p1, p2, p3]) = *self;
        Quad::new(3.0 * (p1 - p0), 3.0 * (p2 - p1), 3.0 * (p3 - p2)).into()
    }

    /// Optimized version of `split_at(0.5)`
    fn split(&self) -> (Self, Self) {
        let Self([p0, p1, p2, p3]) = *self;
        let mid = 0.125 * p0 + 0.375 * p1 + 0.375 * p2 + 0.125 * p3;
        let c0 = Self([
            p0,
            0.5 * p0 + 0.5 * p1,
            0.25 * p0 + 0.5 * p1 + 0.25 * p2,
            mid,
        ]);
        let c1 = Self([
            mid,
            0.25 * p1 + 0.5 * p2 + 0.25 * p3,
            0.5 * p2 + 0.5 * p3,
            p3,
        ]);
        (c0, c1)
    }

    fn split_at(&self, t: Scalar) -> (Self, Self) {
        // https://pomax.github.io/bezierinfo/#matrixsplit
        let Self([p0, p1, p2, p3]) = self;
        let (t1, t_1) = (t, 1.0 - t);
        let (t2, t_2) = (t1 * t1, t_1 * t_1);
        let (t3, t_3) = (t2 * t1, t_2 * t_1);
        let mid = t_3 * p0 + 3.0 * t1 * t_2 * p1 + 3.0 * t2 * t_1 * p2 + t3 * p3;
        let c0 = Self([
            *p0,
            t_1 * p0 + t * p1,
            t_2 * p0 + 2.0 * t * t_1 * p1 + t2 * p2,
            mid,
        ]);
        let c1 = Self([
            mid,
            t_2 * p1 + 2.0 * t * t_1 * p2 + t2 * p3,
            t_1 * p2 + t * p3,
            *p3,
        ]);
        (c0, c1)
    }

    fn cut(&self, a: Scalar, b: Scalar) -> Self {
        // Given curve as C(t) = [1 t t^2 t^3] M C
        // we can change parameter t -> a + (b - a) * t which will produced desired curve
        // it is possible to decompose it as
        //                 ┌                                       ┐
        // ┌             ┐ │  1  a       a^2         a^3           │
        // │ 1 t t^2 t^3 │ │  0  (b - a) 2*a*(b - a) 3*a^2*(b - a) │ = [1 t t^2 t^3] T
        // └             ┘ │  0  0       (b - a)^2   3*a*(b - a)^2 │
        //                 │  0  0       0           (b - a)^3     │
        //                 └                                       ┘
        // we can convert it back to desired curve by C[a, b](t) = [1 t t^2 t^3] C (CI T C) P
        let Self([p0, p1, p2, p3]) = self;
        let ba = b - a;
        #[rustfmt::skip]
        let t = M4x4([
            1.0, a  , a * a       , a * a * a        ,
            0.0, ba , 2.0 * a * ba, 3.0 * a * a * ba ,
            0.0, 0.0, ba * ba     , 3.0 * a * ba * ba,
            0.0, 0.0, 0.0         , ba * ba * ba     ,
        ]);
        #[rustfmt::skip]
        let M4x4([
            m00, m01, m02, m03,
            m10, m11, m12, m13,
            m20, m21, m22, m23,
            m30, m31, m32, m33,
        ]) = CI * t * C;
        let c0 = m00 * p0 + m01 * p1 + m02 * p2 + m03 * p3;
        let c1 = m10 * p0 + m11 * p1 + m12 * p2 + m13 * p3;
        let c2 = m20 * p0 + m21 * p1 + m22 * p2 + m23 * p3;
        let c3 = m30 * p0 + m31 * p1 + m32 * p2 + m33 * p3;
        Self([c0, c1, c2, c3])
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        let Self([p0, p1, p2, p3]) = self;
        let bbox = BBox::new(*p0, *p3).union_opt(init);
        if bbox.contains(*p1) && bbox.contains(*p2) {
            return bbox;
        }
        self.extremities()
            .fold(bbox, |bbox, t| bbox.extend(self.at(t)))
    }

    /// Offset cubic bezier curve with a list of cubic curves
    ///
    /// Offset bezier curve using Tiller-Hanson method. In short, it will just offset
    /// line segment corresponding to control points, then find intersection of this
    /// lines and treat them as new control points.
    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>) {
        cubic_offset_rec(*self, None, dist, out, 0);
    }

    fn reverse(&self) -> Self {
        let Self([p0, p1, p2, p3]) = *self;
        Self([p3, p2, p1, p0])
    }

    fn roots(&self) -> CurveRoots {
        let mut result = CurveRoots::new();
        // curve(t)_y = 0
        let Self([Point([_, y0]), Point([_, y1]), Point([_, y2]), Point([_, y3])]) = *self;
        let a = -y0 + 3.0 * y1 - 3.0 * y2 + y3;
        let b = 3.0 * y0 - 6.0 * y1 + 3.0 * y2;
        let c = -3.0 * y0 + 3.0 * y1;
        let d = y0;
        result.extend(cubic_solve(a, b, c, d).filter(|t| (0.0..=1.0).contains(t)));
        result
    }

    fn extremities(&self) -> CurveExtremities {
        let Self([p0, p1, p2, p3]) = *self;
        let Point([a0, a1]) = -1.0 * p0 + 3.0 * p1 - 3.0 * p2 + 1.0 * p3;
        let Point([b0, b1]) = 2.0 * p0 - 4.0 * p1 + 2.0 * p2;
        let Point([c0, c1]) = -1.0 * p0 + p1;

        // Solve for `curve'(t)_x = 0 || curve'(t)_y = 0`
        quadratic_solve(a0, b0, c0)
            .chain(quadratic_solve(a1, b1, c1))
            .filter(|t| *t >= 0.0 && *t <= 1.0)
            .collect::<CurveExtremities>()
    }
}

pub struct CubicFlattenIter {
    flatness: Scalar,
    cubics: Vec<Cubic>,
}

impl Iterator for CubicFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.cubics.pop() {
                None => {
                    return None;
                }
                Some(cubic) if cubic.flatness() < self.flatness => {
                    let Cubic([p0, _p1, _p2, p3]) = cubic;
                    return Some(Line([p0, p3]));
                }
                Some(cubic) => {
                    let (c0, c1) = cubic.split();
                    self.cubics.push(c1);
                    self.cubics.push(c0);
                }
            }
        }
    }
}

impl From<Quad> for Cubic {
    fn from(quad: Quad) -> Self {
        let Quad([p0, p1, p2]) = quad;
        Self([
            p0,
            (1.0 / 3.0) * p0 + (2.0 / 3.0) * p1,
            (2.0 / 3.0) * p1 + (1.0 / 3.0) * p2,
            p2,
        ])
    }
}

impl FromStr for Cubic {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let segment = Segment::from_str(text)?;
        segment.to_cubic().ok_or_else(|| Error::ConvertionError {
            reason: "first element of the path is not a cubic".to_string(),
        })
    }
}

// -----------------------------------------------------------------------------
// Segment
// -----------------------------------------------------------------------------

/// `Segment` is an enum of either `Line`, `Quad` or `Cubic`
#[derive(Clone, Copy, PartialEq)]
pub enum Segment {
    Line(Line),
    Quad(Quad),
    Cubic(Cubic),
}

impl Segment {
    pub fn ends(&self) -> (Line, Line) {
        match self {
            Segment::Line(line) => line.ends(),
            Segment::Quad(quad) => quad.ends(),
            Segment::Cubic(cubic) => cubic.ends(),
        }
    }

    /// Find intersection between two segments
    ///
    /// This might not be the fastest method possible but works for any two curves.
    /// Divide cuves as long as there is intersection between bounding boxes, if
    /// the intersection is smaller then tolerance we can treat it as an intersection point.
    pub fn intersect(self, other: impl Into<Segment>, tolerance: Scalar) -> Vec<Point> {
        let mut queue = vec![(self, other.into())];
        let mut result = Vec::new();
        while let Some((s0, s1)) = queue.pop() {
            let b0 = s0.bbox(None);
            let b1 = s1.bbox(None);
            match b0.intersect(b1) {
                None => continue,
                Some(b) => {
                    let b0_is_small = b0.width() < tolerance && b0.height() < tolerance;
                    let b1_is_small = b1.width() < tolerance && b1.height() < tolerance;
                    if b0_is_small && b1_is_small {
                        result.push(b.diag().at(0.5));
                    } else {
                        // TODO: can be optimized by spliting only curves with large bbox
                        let (s00, s01) = s0.split_at(0.5);
                        let (s10, s11) = s1.split_at(0.5);
                        queue.push((s00, s10));
                        queue.push((s00, s11));
                        queue.push((s01, s10));
                        queue.push((s01, s11));
                    }
                }
            }
        }
        result
    }

    /// Convert to line if it is a line variant of the segment
    pub fn to_line(&self) -> Option<Line> {
        match self {
            Segment::Line(line) => Some(*line),
            _ => None,
        }
    }

    /// Convert to quad if it is a quad variant of the segment
    pub fn to_quad(&self) -> Option<Quad> {
        match self {
            Segment::Quad(quad) => Some(*quad),
            _ => None,
        }
    }

    /// Convert to cubic if it is a cubic variant of the segment
    pub fn to_cubic(&self) -> Option<Cubic> {
        match self {
            Segment::Cubic(cubic) => Some(*cubic),
            _ => None,
        }
    }

    /// Produce iterator over segments that join to segments with the specified method.
    pub fn line_join(
        self,
        other: Segment,
        stroke_style: StrokeStyle,
    ) -> impl Iterator<Item = Self> {
        let mut result = ArrayIter::<[Option<Segment>; 4]>::new();
        if self.end().is_close_to(other.start()) {
            return result;
        }
        let bevel = Line::new(self.end(), other.start());
        // https://www.w3.org/TR/SVG2/painting.html#LineJoin
        match stroke_style.line_join {
            LineJoin::Bevel => {
                result.push(bevel.into());
            }
            LineJoin::Miter(miter_limit) => {
                let (_, start) = self.ends();
                let (end, _) = other.ends();
                match start.intersect(end) {
                    Some((t0, t1)) if (0.0..=1.0).contains(&t0) && (0.0..=1.0).contains(&t1) => {
                        // ends intersect
                        result.push(bevel.into());
                    }
                    None => result.push(bevel.into()),
                    Some((t, _)) => {
                        let p0 = start.end() - start.start();
                        let p1 = end.start() - end.end();
                        // miter_length = stroke_width / sin(a / 2)
                        // sin(a / 2) = +/- ((1 - cos(a)) / 2).sqrt()
                        let miter_length = p0
                            .cos_between(p1)
                            .map(|c| stroke_style.width / ((1.0 - c) / 2.0).sqrt());
                        match miter_length {
                            Some(miter_length) if miter_length < miter_limit => {
                                let p = start.at(t);
                                result.push(Line::new(start.end(), p).into());
                                result.push(Line::new(p, end.start()).into());
                            }
                            _ => result.push(bevel.into()),
                        }
                    }
                }
            }
            LineJoin::Round => {
                let (_, start) = self.ends();
                let (end, _) = other.ends();
                match start.intersect_point(end) {
                    Some(_) => result.push(bevel.into()),
                    None => {
                        let sweep_flag = start.direction().cross(bevel.direction()) >= 0.0;
                        let radius = stroke_style.width / 2.0;
                        let arc = EllipArc::new_param(
                            start.end(),
                            end.start(),
                            radius,
                            radius,
                            0.0,
                            false,
                            sweep_flag,
                        );
                        match arc {
                            Some(arc) => result.extend(arc.to_cubics().map(Segment::from)),
                            None => result.push(bevel.into()),
                        }
                    }
                }
            }
        }
        result
    }

    /// Produce and iterator over segments that adds caps between two segments
    pub fn line_cap(self, other: Segment, stroke_style: StrokeStyle) -> impl Iterator<Item = Self> {
        let mut result = ArrayIter::<[Option<Segment>; 4]>::new();
        if self.end().is_close_to(other.start()) {
            return result;
        }
        let butt = Line::new(self.end(), other.start());
        match stroke_style.line_cap {
            LineCap::Butt => result.push(butt.into()),
            LineCap::Square => {
                let (_, from) = self.ends();
                if let Some(tang) = from.direction().normalize() {
                    let l0 = Line::new(self.end(), self.end() + stroke_style.width / 2.0 * tang);
                    result.push(l0.into());
                    let l1 = Line::new(l0.end(), l0.end() + butt.direction());
                    result.push(l1.into());
                    let l2 = Line::new(l1.end(), other.start());
                    result.push(l2.into());
                }
            }
            LineCap::Round => {
                let stroke_style = StrokeStyle {
                    line_join: LineJoin::Round,
                    ..stroke_style
                };
                result.extend(self.line_join(other, stroke_style));
            }
        }
        result
    }
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Segment::Line(line) => line.fmt(f),
            Segment::Quad(quad) => quad.fmt(f),
            Segment::Cubic(cubic) => cubic.fmt(f),
        }
    }
}

impl FromStr for Segment {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let path = Path::from_str(text)?;
        path.subpaths()
            .get(0)
            .map(|sp| sp.first())
            .ok_or_else(|| Error::ConvertionError {
                reason: "Empty path can not be converted to a segment".to_string(),
            })
    }
}

impl Curve for Segment {
    fn flatness(&self) -> Scalar {
        match self {
            Segment::Line(line) => line.flatness(),
            Segment::Quad(quad) => quad.flatness(),
            Segment::Cubic(cubic) => cubic.flatness(),
        }
    }

    fn transform(&self, tr: Transform) -> Self {
        match self {
            Segment::Line(line) => line.transform(tr).into(),
            Segment::Quad(quad) => quad.transform(tr).into(),
            Segment::Cubic(cubic) => cubic.transform(tr).into(),
        }
    }

    fn start(&self) -> Point {
        match self {
            Segment::Line(line) => line.start(),
            Segment::Quad(quad) => quad.start(),
            Segment::Cubic(cubic) => cubic.start(),
        }
    }

    fn end(&self) -> Point {
        match self {
            Segment::Line(line) => line.end(),
            Segment::Quad(quad) => quad.end(),
            Segment::Cubic(cubic) => cubic.end(),
        }
    }

    fn at(&self, t: Scalar) -> Point {
        match self {
            Segment::Line(line) => line.at(t),
            Segment::Quad(quad) => quad.at(t),
            Segment::Cubic(cubic) => cubic.at(t),
        }
    }

    fn deriv(&self) -> Segment {
        match self {
            Segment::Line(line) => line.deriv(),
            Segment::Quad(quad) => quad.deriv(),
            Segment::Cubic(cubic) => cubic.deriv(),
        }
    }

    fn split_at(&self, t: Scalar) -> (Self, Self) {
        match self {
            Segment::Line(line) => {
                let (l0, l1) = line.split_at(t);
                (l0.into(), l1.into())
            }
            Segment::Quad(quad) => {
                let (q0, q1) = quad.split_at(t);
                (q0.into(), q1.into())
            }
            Segment::Cubic(cubic) => {
                let (c0, c1) = cubic.split_at(t);
                (c0.into(), c1.into())
            }
        }
    }

    fn cut(&self, a: Scalar, b: Scalar) -> Self {
        match self {
            Segment::Line(line) => line.cut(a, b).into(),
            Segment::Quad(quad) => quad.cut(a, b).into(),
            Segment::Cubic(cubic) => cubic.cut(a, b).into(),
        }
    }

    fn bbox(&self, init: Option<BBox>) -> BBox {
        match self {
            Segment::Line(line) => line.bbox(init),
            Segment::Quad(quad) => quad.bbox(init),
            Segment::Cubic(cubic) => cubic.bbox(init),
        }
    }

    fn offset(&self, dist: Scalar, out: &mut impl Extend<Segment>) {
        match self {
            Segment::Line(line) => line.offset(dist, out),
            Segment::Quad(quad) => quad.offset(dist, out),
            Segment::Cubic(cubic) => cubic.offset(dist, out),
        }
    }

    fn reverse(&self) -> Self {
        match self {
            Segment::Line(line) => line.reverse().into(),
            Segment::Quad(quad) => quad.reverse().into(),
            Segment::Cubic(cubic) => cubic.reverse().into(),
        }
    }

    fn roots(&self) -> CurveRoots {
        match self {
            Segment::Line(line) => line.roots(),
            Segment::Quad(quad) => quad.roots(),
            Segment::Cubic(cubic) => cubic.roots(),
        }
    }

    fn extremities(&self) -> CurveExtremities {
        match self {
            Segment::Line(line) => line.extremities(),
            Segment::Quad(quad) => quad.extremities(),
            Segment::Cubic(cubic) => cubic.extremities(),
        }
    }
}

impl From<Line> for Segment {
    fn from(line: Line) -> Self {
        Self::Line(line)
    }
}

impl From<Quad> for Segment {
    fn from(quad: Quad) -> Self {
        Self::Quad(quad)
    }
}

impl From<Cubic> for Segment {
    fn from(cubic: Cubic) -> Self {
        Self::Cubic(cubic)
    }
}

// -----------------------------------------------------------------------------
// Bezier curve offsetting helpers
// -----------------------------------------------------------------------------

/// Offset line to the distance.
pub(crate) fn line_offset(line: Line, dist: Scalar) -> Option<Line> {
    let Line([p0, p1]) = line;
    let offset = dist * (p1 - p0).normal().normalize()?;
    Some(Line::new(p0 + offset, p1 + offset))
}

/// Offset polyline specified by points `ps`.
///
/// Implementation correctly handles repeated points.
/// False result indicates that all points are close to each other
/// and it is not possible to offset them.
fn polyline_offset(ps: &mut [Point], dist: Scalar) -> bool {
    if ps.is_empty() {
        return true;
    }
    let mut prev: Option<Line> = None;
    let mut index = 0;
    loop {
        // find identical points repeats
        let mut repeats = 1;
        for i in index..ps.len() - 1 {
            if !ps[i].is_close_to(ps[i + 1]) {
                break;
            }
            repeats += 1;
        }
        if index + repeats >= ps.len() {
            break;
        }
        index += repeats;
        // offset line
        let next = line_offset(Line::new(ps[index - 1], ps[index]), dist)
            .expect("polyline implementation error");
        // find where to move repeated points
        let point = match prev {
            None => next.start(),
            Some(prev) => match prev.intersect(next) {
                Some((t, _)) => prev.at(t),
                None => {
                    // TODO: not sure what to do especially for up/down move
                    next.start()
                }
            },
        };
        // move repeats
        for p in ps.iter_mut().take(index).skip(index - repeats) {
            *p = point;
        }
        prev = Some(next);
    }
    // handle tail points
    match prev {
        None => {
            // all points are close to each other, can not offset.
            false
        }
        Some(prev) => {
            for p in ps.iter_mut().skip(index) {
                *p = prev.end();
            }
            true
        }
    }
}

/// Determine if quad curve needs splitting before offsetting.
fn quad_offset_should_split(quad: Quad) -> bool {
    let Quad([p0, p1, p2]) = quad;
    // split if angle is sharp
    if (p0 - p1).dot(p2 - p1) > 0.0 {
        return true;
    }
    // distance between center mass and midpoint of a cruve,
    // should be bigger then 0.1 of the bounding box diagonal
    let c_mass = (p0 + p1 + p2) / 3.0;
    let c_mid = quad.at(0.5);
    let dist = (c_mass - c_mid).length();
    let bbox_diag = quad.bbox(None).diag().length();
    bbox_diag * 0.1 < dist
}

/// Recursive quad offset.
fn quad_offset_rec(quad: Quad, dist: Scalar, out: &mut impl Extend<Segment>, depth: usize) {
    if quad_offset_should_split(quad) && depth < 3 {
        let (c0, c1) = quad.split_at(0.5);
        quad_offset_rec(c0, dist, out, depth + 1);
        quad_offset_rec(c1, dist, out, depth + 1);
    } else {
        let mut points = quad.points();
        if polyline_offset(&mut points, dist) {
            out.extend(Some(Quad(points).into()));
        }
    }
}

/// Recursive cubic offset.
fn cubic_offset_rec(
    cubic: Cubic,
    last: Option<Segment>,
    dist: Scalar,
    out: &mut impl Extend<Segment>,
    depth: usize,
) -> Option<Segment> {
    if cubic_offset_should_split(cubic) && depth < 3 {
        let (c0, c1) = cubic.split();
        let last = cubic_offset_rec(c0, last, dist, out, depth + 1);
        return cubic_offset_rec(c1, last, dist, out, depth + 1);
    }
    let mut points = cubic.points();
    if polyline_offset(&mut points, dist) {
        let result: Segment = Cubic(points).into();
        // there could a disconnect between offset curvese.
        // For example M0,0 C10,5 0,5 10,0.
        if let Some(last) = last {
            if !last.end().is_close_to(result.start()) {
                let stroke_style = StrokeStyle {
                    width: dist * 2.0,
                    line_join: LineJoin::Round,
                    line_cap: LineCap::Round,
                };
                out.extend(last.line_join(result, stroke_style));
            }
        }
        out.extend(Some(result));
        Some(result)
    } else {
        last
    }
}

/// Determine if cubic curve needs splitting before offsetting.
fn cubic_offset_should_split(cubic: Cubic) -> bool {
    let Cubic([p0, p1, p2, p3]) = cubic;
    // angle(p3 - p0, p2 - p1) > 90 or < -90
    if (p3 - p0).dot(p2 - p1) < 0.0 {
        return true;
    }
    // control points should be on the same side of the baseline.
    let a0 = (p3 - p0).cross(p1 - p0);
    let a1 = (p3 - p0).cross(p2 - p0);
    if a0 * a1 < 0.0 {
        return true;
    }
    // distance between center mass and midpoint of a cruve,
    // should be bigger then 0.1 of the bounding box diagonal
    let c_mass = (p0 + p1 + p2 + p3) / 4.0;
    let c_mid = cubic.at(0.5);
    let dist = (c_mass - c_mid).length();
    let bbox_diag = cubic.bbox(None).diag().length();
    bbox_diag * 0.1 < dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    fn test_polyline_offset() {
        let dist = -(2.0 as Scalar).sqrt();

        let p0 = Point::new(1.0, 0.0);
        let r0 = Point::new(0.0, 1.0);

        let p1 = Point::new(2.0, 1.0);
        let r1 = Point::new(2.0, 3.0);

        let p2 = Point::new(3.0, 0.0);
        let r2 = Point::new(4.0, 1.0);

        // basic
        let mut ps = [p0, p1, p2];
        assert!(polyline_offset(&mut ps, dist));
        assert_eq!(&ps, &[r0, r1, r2]);

        // repeat at start
        let mut ps = [p0, p0, p1, p2];
        assert!(polyline_offset(&mut ps, dist));
        assert_eq!(&ps, &[r0, r0, r1, r2]);

        // repeat at start
        let mut ps = [p0, p1, p2, p2];
        assert!(polyline_offset(&mut ps, dist));
        assert_eq!(&ps, &[r0, r1, r2, r2]);

        // repeat in the middle
        let mut ps = [p0, p1, p1, p2];
        assert!(polyline_offset(&mut ps, dist));
        assert_eq!(&ps, &[r0, r1, r1, r2]);

        // all points are close to each other, can not offset
        let mut ps = [p0, p0, p0];
        assert!(!polyline_offset(&mut ps, dist));

        // splitted single line
        let mut ps = [p0, p1, Point::new(3.0, 2.0)];
        assert!(polyline_offset(&mut ps, dist));
        assert_eq!(&ps, &[r0, Point::new(1.0, 2.0), r1]);

        // four points
        let mut ps = [p0, p1, p2, Point::new(2.0, -1.0)];
        assert!(polyline_offset(&mut ps, dist));
        assert_eq!(&ps, &[r0, r1, Point::new(5.0, 0.0), Point::new(3.0, -2.0)]);
    }

    #[test]
    fn test_roots() {
        let l = Line::new((0.0, -1.0), (2.0, 1.0));
        assert_eq!(l.roots().collect::<Vec<_>>(), vec![0.5]);

        let q = Quad::new((0.0, -2.0), (7.0, 6.0), (6.0, -4.0));
        assert_eq!(
            q.roots().collect::<Vec<_>>(),
            vec![0.73841681234051, 0.15047207654837882]
        );

        let c = Cubic::new((0.0, -2.0), (2.0, 4.0), (4.0, -3.0), (9.0, 1.0));
        assert_eq!(
            c.roots().collect::<Vec<_>>(),
            vec![0.8812186869024836, 0.1627575589800928, 0.5810237541174236]
        );

        let c: Cubic = "M8,-1 C1,3 6,-3 9,1".parse().unwrap();
        assert_eq!(
            c.roots().collect::<Vec<_>>(),
            vec![0.8872983346207419, 0.11270166537925835, 0.4999999999999995]
        );
    }

    #[test]
    fn test_curve_matrices() {
        #[rustfmt::skip]
        let i3 = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ];
        let M3x3(q) = Q * QI;
        assert_eq!(i3.len(), q.len());
        for (v0, v1) in i3.iter().zip(q.iter()) {
            assert_approx_eq!(v0, v1);
        }

        #[rustfmt::skip]
        let i4 = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let M4x4(c) = C * CI;
        assert_eq!(i4.len(), c.len());
        for (v0, v1) in i4.iter().zip(c.iter()) {
            assert_approx_eq!(v0, v1);
        }
    }

    #[test]
    fn test_ends() {
        let p0 = Point::new(1.0, 0.0);
        let p1 = Point::new(2.0, 1.0);
        let p2 = Point::new(3.0, 0.0);
        let p3 = Point::new(2.0, 0.0);

        let c = Cubic::new(p0, p1, p2, p3);
        let (start, end) = c.ends();
        assert_eq!(start, Line::new(p0, p1));
        assert_eq!(end, Line::new(p2, p3));

        let c = Cubic::new(p0, p0, p1, p2);
        let (start, end) = c.ends();
        assert_eq!(start, Line::new(p0, p1));
        assert_eq!(end, Line::new(p1, p2));

        let c = Cubic::new(p0, p1, p2, p2);
        let (start, end) = c.ends();
        assert_eq!(start, Line::new(p0, p1));
        assert_eq!(end, Line::new(p1, p2));

        let q = Quad::new(p0, p1, p2);
        let (start, end) = q.ends();
        assert_eq!(start, Line::new(p0, p1));
        assert_eq!(end, Line::new(p1, p2));

        let q = Quad::new(p0, p0, p1);
        let (start, end) = q.ends();
        assert_eq!(start, Line::new(p0, p1));
        assert_eq!(end, Line::new(p0, p1));

        let q = Quad::new(p0, p1, p1);
        let (start, end) = q.ends();
        assert_eq!(start, Line::new(p0, p1));
        assert_eq!(end, Line::new(p0, p1));
    }

    #[test]
    fn test_split() {
        let q = Quad::new((0.0, 0.0), (8.0, 5.0), (4.0, 0.0));
        let (ql, qr) = q.split();
        assert_eq!((ql, qr), q.split_at(0.5));
        assert_eq!(ql, q.cut(0.0, 0.5));
        assert_eq!(qr, q.cut(0.5, 1.0));

        let c = Cubic::new((3.0, 7.0), (2.0, 8.0), (0.0, 3.0), (6.0, 5.0));
        let (cl, cr) = c.split();
        assert_eq!((cl, cr), c.split_at(0.5));
        assert_eq!(cl, c.cut(0.0, 0.5));
        assert_eq!(cr, c.cut(0.5, 1.0));
    }

    #[test]
    fn test_bbox() {
        let b0 = BBox::new(Point::new(2.0, 2.0), Point::new(4.0, 4.0));
        let b1 = b0.extend(Point::new(1.0, 3.0));
        assert!(b1.min().is_close_to(Point::new(1.0, 2.0)));
        assert!(b1.max().is_close_to(b0.max()));
        let b2 = b1.extend(Point::new(5.0, 3.0));
        assert!(b2.min().is_close_to(b1.min()));
        assert!(b2.max().is_close_to(Point::new(5.0, 4.0)));
        let b3 = b2.extend(Point::new(3.0, 1.0));
        assert!(b3.min().is_close_to(Point::new(1.0, 1.0)));
        assert!(b3.max().is_close_to(b2.max()));
        let b4 = b3.extend(Point::new(3.0, 5.0));
        assert!(b4.min().is_close_to(b3.min()));
        assert!(b4.max().is_close_to(Point::new(5.0, 5.0)));

        let cubic = Cubic::new((106.0, 0.0), (0.0, 100.0), (382.0, 216.0), (324.0, 14.0));
        let bbox = cubic.bbox(None);
        assert_approx_eq!(bbox.x(), 87.308, 0.001);
        assert_approx_eq!(bbox.y(), 0.0, 0.001);
        assert_approx_eq!(bbox.width(), 242.724, 0.001);
        assert_approx_eq!(bbox.height(), 125.140, 0.001);

        let quad = Quad::new((30.0, 90.0), (220.0, 200.0), (120.0, 50.0));
        let bbox = quad.bbox(None);
        assert_approx_eq!(bbox.x(), 30.0, 0.001);
        assert_approx_eq!(bbox.y(), 50.0, 0.001);
        assert_approx_eq!(bbox.width(), 124.483, 0.001);
        assert_approx_eq!(bbox.height(), 86.538, 0.001);

        let cubic = Cubic::new((0.0, 0.0), (10.0, -3.0), (-4.0, -3.0), (6.0, 0.0));
        let bbox = cubic.bbox(None);
        assert_approx_eq!(bbox.x(), 0.0);
        assert_approx_eq!(bbox.y(), -2.25);
        assert_approx_eq!(bbox.width(), 6.0);
        assert_approx_eq!(bbox.height(), 2.25);
    }
}
