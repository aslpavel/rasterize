use crate::{BBox, Cubic, Curve, CurveFlattenIter, Line, Point, Scalar, Transform, PI};
use std::fmt;

/// Elliptical Arc
#[derive(Clone, Copy, PartialEq)]
pub struct EllipArc {
    /// center of the ellipse
    center: Point,
    /// radius along x-axis before the rotation
    rx: Scalar,
    /// radius along y-axis before the rotation
    ry: Scalar,
    /// rotation
    phi: Scalar,
    /// angular start
    eta: Scalar,
    /// angular size
    eta_delta: Scalar,
}

impl fmt::Debug for EllipArc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Arc center:{:?} radius:{:?} phi:{:.3?} eta:{:.3?} eta_delta:{:.3?}",
            self.center,
            Point([self.rx, self.ry]),
            self.phi,
            self.eta,
            self.eta_delta
        )
    }
}

impl EllipArc {
    /// Convert arc from SVG arguments to parametric curve
    ///
    /// This code mostly comes from arc implementation notes from svg sepc
    /// (Arc to Parametric)[https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes]
    pub fn new_param(
        src: Point,
        dst: Point,
        rx: Scalar,
        ry: Scalar,
        x_axis_rot: Scalar,
        large_flag: bool,
        sweep_flag: bool,
    ) -> Option<Self> {
        let rx = rx.abs();
        let ry = ry.abs();
        let phi = x_axis_rot * PI / 180.0;

        // Eq 5.1
        let Point([x1, y1]) = Transform::default().rotate(-phi).apply(0.5 * (src - dst));
        // scale/normalize radii
        let s = (x1 / rx).powi(2) + (y1 / ry).powi(2);
        let (rx, ry) = if s > 1.0 {
            let s = s.sqrt();
            (rx * s, ry * s)
        } else {
            (rx, ry)
        };
        // Eq 5.2
        let sq = ((rx * ry).powi(2) / ((rx * y1).powi(2) + (ry * x1).powi(2)) - 1.0)
            .max(0.0)
            .sqrt();
        let sq = if large_flag == sweep_flag { -sq } else { sq };
        let center = sq * Point([rx * y1 / ry, -ry * x1 / rx]);
        let Point([cx, cy]) = center;
        // Eq 5.3 convert center to initail coordinates
        let center = Transform::default().rotate(phi).apply(center) + 0.5 * (dst + src);
        // Eq 5.5-6
        let v0 = Point([1.0, 0.0]);
        let v1 = Point([(x1 - cx) / rx, (y1 - cy) / ry]);
        let v2 = Point([(-x1 - cx) / rx, (-y1 - cy) / ry]);
        // initial angle
        let eta = v0.angle_between(v1)?;
        //delta angle to be covered when t changes from 0..1
        let eta_delta = v1.angle_between(v2)?.rem_euclid(2.0 * PI);
        let eta_delta = if !sweep_flag && eta_delta > 0.0 {
            eta_delta - 2.0 * PI
        } else if sweep_flag && eta_delta < 0.0 {
            eta_delta + 2.0 * PI
        } else {
            eta_delta
        };

        Some(Self {
            center,
            rx,
            ry,
            phi,
            eta,
            eta_delta,
        })
    }

    pub fn at(&self, t: Scalar) -> Point {
        let (angle_sin, angle_cos) = (self.eta + t * self.eta_delta).sin_cos();
        let point = Point([self.rx * angle_cos, self.ry * angle_sin]);
        Transform::default().rotate(self.phi).apply(point) + self.center
    }

    pub fn start(&self) -> Point {
        self.at(0.0)
    }

    pub fn end(&self) -> Point {
        self.at(1.0)
    }

    pub fn bbox(&self, init: Option<BBox>) -> BBox {
        EllipArcCubicIter::new(*self)
            .fold(init, |bbox, cubic| Some(cubic.bbox(bbox)))
            .expect("EllipArcCubicIter is empty")
    }

    pub fn reverse(&self) -> Self {
        Self {
            center: self.center,
            rx: self.rx,
            ry: self.ry,
            phi: self.phi,
            eta: self.eta + self.eta_delta,
            eta_delta: -self.eta_delta,
        }
    }

    /// Convert elliptic arc to an iterator over Cubic segments
    pub fn to_cubics(&self) -> EllipArcCubicIter {
        EllipArcCubicIter::new(*self)
    }

    /// Convert elliptic arc to an iterator over line segments with desired flatness
    pub fn flatten(&self, tr: Transform, flatness: Scalar) -> EllipArcFlattenIter {
        EllipArcFlattenIter::new(*self, tr, flatness)
    }
}

/// Approximate arc with a sequnce of cubic bezier curves
///
/// [Drawing an elliptical arc using polylines, quadratic or cubic Bezier curves]
/// (http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf)
/// [Approximating Arcs Using Cubic Bézier Curves]
/// (https://www.joecridge.me/content/pdf/bezier-arcs.pdf)
///
/// We are using following formula to split arc segment from `eta_1` to `eta_2`
/// to achieve good approximation arc is split in segments smaller then `pi / 2`.
///     P0 = A(eta_1)
///     P1 = P0 + alpha * A'(eta_1)
///     P2 = P3 - alpha * A'(eta_2)
///     P3 = A(eta_2)
/// where
///     A - arc parametrized by angle
///     A' - derivative of arc parametrized by angle
///     eta_1 = eta
///     eta_2 = eta + eta_delta
///     alpha = sin(eta_2 - eta_1) * (sqrt(4 + 3 * tan((eta_2 - eta_1) / 2) ** 2) - 1) / 3
pub struct EllipArcCubicIter {
    arc: EllipArc,
    phi_tr: Transform,
    segment_delta: Scalar,
    segment_index: Scalar,
    segment_count: Scalar,
}

impl EllipArcCubicIter {
    fn new(arc: EllipArc) -> Self {
        let phi_tr = Transform::default().rotate(arc.phi);
        let segment_max_angle = PI / 2.0; // maximum `eta_delta` of a segment
        let segment_count = (arc.eta_delta.abs() / segment_max_angle).ceil();
        let segment_delta = arc.eta_delta / segment_count;
        Self {
            arc,
            phi_tr,
            segment_delta,
            segment_index: 0.0,
            segment_count: segment_count - 1.0,
        }
    }

    fn at(&self, alpha: Scalar) -> (Point, Point) {
        let (sin, cos) = alpha.sin_cos();
        let at = self
            .phi_tr
            .apply(Point([self.arc.rx * cos, self.arc.ry * sin]))
            + self.arc.center;
        let at_deriv = self
            .phi_tr
            .apply(Point([-self.arc.rx * sin, self.arc.ry * cos]));
        (at, at_deriv)
    }
}

impl Iterator for EllipArcCubicIter {
    type Item = Cubic;

    fn next(&mut self) -> Option<Self::Item> {
        if self.segment_index > self.segment_count {
            return None;
        }
        let eta_1 = self.arc.eta + self.segment_delta * self.segment_index;
        let eta_2 = eta_1 + self.segment_delta;
        self.segment_index += 1.0;

        let sq = (4.0 + 3.0 * ((eta_2 - eta_1) / 2.0).tan().powi(2)).sqrt();
        let alpha = (eta_2 - eta_1).sin() * (sq - 1.0) / 3.0;
        let (p0, d0) = self.at(eta_1);
        let (p3, d3) = self.at(eta_2);
        let p1 = p0 + alpha * d0;
        let p2 = p3 - alpha * d3;
        Some(Cubic([p0, p1, p2, p3]))
    }
}

pub struct EllipArcFlattenIter {
    tr: Transform,
    flatness: Scalar,
    cubics: EllipArcCubicIter,
    cubic: Option<CurveFlattenIter>,
}

impl EllipArcFlattenIter {
    fn new(arc: EllipArc, tr: Transform, flatness: Scalar) -> Self {
        Self {
            tr,
            flatness,
            cubics: arc.to_cubics(),
            cubic: None,
        }
    }
}

impl Iterator for EllipArcFlattenIter {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.cubic.as_mut().and_then(Iterator::next) {
                line @ Some(_) => return line,
                None => self.cubic = Some(self.cubics.next()?.flatten(self.tr, self.flatness)),
            }
        }
    }
}
