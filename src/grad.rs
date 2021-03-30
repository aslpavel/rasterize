use crate::{LinColor, Paint, Point, Scalar, Transform, Units};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradSpread {
    Pad,
    Repeat,
    Reflect,
}

impl GradSpread {
    pub fn at(&self, t: Scalar) -> Scalar {
        // TODO: make sure that branch predicution eliminates this match
        use GradSpread::*;
        match self {
            Pad => t,
            Repeat => t.rem_euclid(1.0),
            Reflect => ((t + 1.0).rem_euclid(2.0) - 1.0).abs(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradStop {
    pub position: Scalar,
    pub color: LinColor,
}

impl GradStop {
    pub fn new(position: Scalar, color: LinColor) -> Self {
        Self { position, color }
    }
}

#[derive(Debug, Clone)]
pub struct GradStops {
    stops: Vec<GradStop>,
}

impl GradStops {
    pub fn new(mut stops: Vec<GradStop>) -> Self {
        stops.sort_by(|s0, s1| {
            s0.position
                .partial_cmp(&s1.position)
                .unwrap_or(Ordering::Greater)
        });
        if stops.is_empty() {
            stops.push(GradStop {
                position: 0.0,
                color: LinColor::new(0.0, 0.0, 0.0, 1.0),
            });
        }
        Self { stops }
    }

    fn convert_to_srgb(&mut self) {
        for stop in self.stops.iter_mut() {
            stop.color = stop.color.into_srgb()
        }
    }
}

impl GradStops {
    fn at(&self, t: Scalar) -> LinColor {
        let index = self.stops.binary_search_by(|stop| {
            if stop.position < t {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        let index = match index {
            Ok(index) => index,
            Err(index) => index,
        };
        let size = self.stops.len();
        if index == 0 {
            self.stops[index].color
        } else if index == size {
            self.stops[size - 1].color
        } else {
            let c0 = self.stops[index - 1].color;
            let c1 = self.stops[index].color;
            c0.lerp(c1, t)
        }
    }
}

impl From<Vec<GradStop>> for GradStops {
    fn from(stops: Vec<GradStop>) -> Self {
        Self::new(stops)
    }
}

#[derive(Debug)]
pub struct GradLinear {
    stops: GradStops,
    units: Units,
    linear: bool,
    start: Point,
    end: Point,
    spread: GradSpread,
    tr: Transform,
    // precomputed value equal to `(end - start) / |end - start| ^ 2`
    dir: Point,
}

impl GradLinear {
    pub fn new(
        stops: impl Into<GradStops>,
        units: Units,
        linear: bool,
        start: impl Into<Point>,
        end: impl Into<Point>,
        spread: GradSpread,
        tr: Transform,
    ) -> Self {
        let start = start.into();
        let end = end.into();
        let mut stops = stops.into();
        if !linear {
            stops.convert_to_srgb();
        }
        let dir = end - start;
        Self {
            stops,
            units,
            linear,
            start,
            end,
            dir: dir / dir.dot(dir),
            spread,
            tr,
        }
    }
}

impl Paint for GradLinear {
    fn at(&self, point: Point) -> LinColor {
        // t = (point - start).dot(end - start) / |end - start| ^ 2
        let t = (point - self.start).dot(self.dir);
        let color = self.stops.at(t);
        if self.linear {
            color
        } else {
            color.into_linear()
        }
    }

    fn transform(&self) -> Transform {
        self.tr
    }

    fn units(&self) -> Option<Units> {
        Some(self.units)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    fn test_spread() {
        use GradSpread::*;
        assert_approx_eq!(Reflect.at(0.3), 0.3, 1e-6);
        assert_approx_eq!(Reflect.at(-0.3), 0.3, 1e-6);
        assert_approx_eq!(Reflect.at(1.3), 0.7, 1e-6);
        assert_approx_eq!(Reflect.at(-1.3), 0.7, 1e-6);

        assert_approx_eq!(Repeat.at(0.3), 0.3);
        assert_approx_eq!(Repeat.at(-0.3), 0.7);
    }
}
