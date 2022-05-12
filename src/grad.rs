use crate::{
    utils::quadratic_solve, LinColor, Paint, Point, Scalar, SvgParserError, Transform, Units,
};
use serde::{de, Deserialize, Serialize};
use std::{borrow::Cow, cmp::Ordering};

/// Gradient spread logic for the parameter smaller than 0 and greater than 1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GradSpread {
    /// Use the same colors as the edge of the gradient
    Pad,
    /// Repeat gradient
    Repeat,
    /// Repeat gradient but alternate reflected and non reflected versions
    Reflect,
}

impl GradSpread {
    /// Map gradient parameter value to the range of [0, 1]
    pub fn at(&self, t: Scalar) -> Scalar {
        match self {
            GradSpread::Pad => t,
            GradSpread::Repeat => t.rem_euclid(1.0),
            GradSpread::Reflect => ((t + 1.0).rem_euclid(2.0) - 1.0).abs(),
        }
    }
}

impl Default for GradSpread {
    fn default() -> Self {
        Self::Pad
    }
}

/// Specifies color at a particular parameter offset of the gradient
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

impl Serialize for GradStop {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let color = self.color.to_string();
        (self.position, color).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GradStop {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (position, color): (Scalar, Cow<'de, str>) = Deserialize::deserialize(deserializer)?;
        Ok(Self {
            position,
            color: color.parse::<LinColor>().map_err(de::Error::custom)?,
        })
    }
}

/// List of all `GradStop` in the gradient
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
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

    fn convert_to_linear(&mut self) {
        for stop in self.stops.iter_mut() {
            stop.color = stop.color.into_linear()
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
            let p0 = &self.stops[index - 1];
            let p1 = &self.stops[index];
            let ratio = (t - p0.position) / (p1.position - p0.position);
            p0.color.lerp(p1.color, ratio)
        }
    }
}

impl From<Vec<GradStop>> for GradStops {
    fn from(stops: Vec<GradStop>) -> Self {
        Self::new(stops)
    }
}

/// Linear Gradient
#[derive(Debug, Clone, PartialEq)]
pub struct GradLinear {
    units: Units,
    linear_colors: bool,
    spread: GradSpread,
    tr: Transform,
    start: Point,
    end: Point,
    // precomputed value equal to `(end - start) / |end - start| ^ 2`
    dir: Point,
    stops: GradStops,
}

impl GradLinear {
    pub const GRAD_TYPE: &'static str = "linear-gradient";

    pub fn new(
        stops: impl Into<GradStops>,
        units: Units,
        linear_colors: bool,
        spread: GradSpread,
        tr: Transform,
        start: impl Into<Point>,
        end: impl Into<Point>,
    ) -> Self {
        let start = start.into();
        let end = end.into();
        let mut stops = stops.into();
        if !linear_colors {
            stops.convert_to_srgb();
        }
        let dir = end - start;
        Self {
            stops,
            units,
            linear_colors,
            spread,
            tr,
            start,
            end,
            dir: dir / dir.dot(dir),
        }
    }

    /// Construct linear gradient from JSON value
    pub fn from_json(value: serde_json::Value) -> Result<Self, SvgParserError> {
        let value: GradLinearSerialize = serde_json::from_value(value)?;
        Ok(value.into())
    }
}

impl Paint for GradLinear {
    fn at(&self, point: Point) -> LinColor {
        // t = (point - start).dot(end - start) / |end - start| ^ 2
        let t = (point - self.start).dot(self.dir);
        let color = self.stops.at(self.spread.at(t));
        if self.linear_colors {
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

    fn to_json(&self) -> Result<serde_json::Value, SvgParserError> {
        let ser: GradLinearSerialize = self.clone().into();
        Ok(serde_json::to_value(ser)?)
    }
}

impl Serialize for GradLinear {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        GradLinearSerialize::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GradLinear {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(GradLinearSerialize::deserialize(deserializer)?.into())
    }
}

/// Intermediate type for linear gradient (de)serialization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GradLinearSerialize {
    #[serde(rename = "type")]
    grad_type: String,
    #[serde(default, skip_serializing_if = "crate::utils::is_default")]
    units: Units,
    #[serde(default, skip_serializing_if = "crate::utils::is_default")]
    linear_colors: bool,
    #[serde(default, skip_serializing_if = "crate::utils::is_default")]
    spread: GradSpread,
    #[serde(
        with = "crate::utils::serde_from_str",
        default,
        skip_serializing_if = "crate::utils::is_default"
    )]
    tr: Transform,
    start: Point,
    end: Point,
    stops: GradStops,
}

impl From<GradLinear> for GradLinearSerialize {
    fn from(mut grad: GradLinear) -> Self {
        if !grad.linear_colors {
            grad.stops.convert_to_linear();
        }
        GradLinearSerialize {
            grad_type: GradLinear::GRAD_TYPE.into(),
            units: grad.units,
            linear_colors: grad.linear_colors,
            spread: grad.spread,
            tr: grad.tr,
            start: grad.start,
            end: grad.end,
            stops: grad.stops,
        }
    }
}

impl From<GradLinearSerialize> for GradLinear {
    fn from(value: GradLinearSerialize) -> Self {
        Self::new(
            value.stops,
            value.units,
            value.linear_colors,
            value.spread,
            value.tr,
            value.start,
            value.end,
        )
    }
}

/// Radial Gradient
#[derive(Debug, Clone, PartialEq)]
pub struct GradRadial {
    units: Units,
    linear_colors: bool,
    spread: GradSpread,
    tr: Transform,
    center: Point,
    radius: Scalar,
    fcenter: Point,
    fradius: Scalar,
    stops: GradStops,
}

impl GradRadial {
    pub const GRAD_TYPE: &'static str = "radial-gradient";

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stops: impl Into<GradStops>,
        units: Units,
        linear_colors: bool,
        spread: GradSpread,
        tr: Transform,
        center: impl Into<Point>,
        radius: Scalar,
        fcenter: impl Into<Point>,
        fradius: Scalar,
    ) -> Self {
        let center = center.into();
        let fcenter = fcenter.into();
        let mut stops = stops.into();
        if !linear_colors {
            stops.convert_to_srgb();
        }
        Self {
            stops,
            units,
            linear_colors,
            spread,
            tr,
            center,
            radius,
            fcenter,
            fradius,
        }
    }

    /// Construct radial gradient from JSON value
    pub fn from_json(value: serde_json::Value) -> Result<Self, SvgParserError> {
        let value: GradRadialSerialize = serde_json::from_value(value)?;
        Ok(value.into())
    }

    /// Calculate gradient offset at a given point
    fn offset(&self, point: Point) -> Option<Scalar> {
        // Two circle gradient is an interpolation between two circles (fc, fr) and (c, r),
        // with center `c(t) = (1 - t) * fc + t * c`, and radius `r(t) = (1 - t) * fr + t * r`.
        // If we have a pixel with coordinates `p`, we should solve equation for it
        // `|| c(t) - p || = r(t)` and pick solution corresponding to bigger radius.
        //
        // Solving this equation for `t`:
        //```
        //     || c(t) - p || = r(t)  -> At² - 2Bt + C = 0
        // where:
        //
        //     cd = c - fc
        //     pd = p - fc
        //     rd = r - fr
        //     A = cdx ^ 2 + cdy ^ 2 - rd ^ 2
        //     B = pdx * cdx + pdy * cdy + fradius * rd
        //     C = pdx ^2 + pdy ^ 2 - fradius ^ 2
        // results in:
        //     t = (B +/- (B ^ 2 - A * C).sqrt()) / A
        //```
        // [reference]: https://cgit.freedesktop.org/pixman/tree/pixman/pixman-radial-gradient.c

        let cd = self.center - self.fcenter;
        let pd = point - self.fcenter;
        let rd = self.radius - self.fradius;

        let a = cd.dot(cd) - rd * rd;
        let b = -2.0 * (cd.dot(pd) + self.fradius * rd);
        let c = pd.dot(pd) - self.fradius * self.fradius;

        match quadratic_solve(a, b, c).into_array() {
            [Some(t0), Some(t1)] => Some(t0.max(t1)),
            [Some(t), None] | [None, Some(t)] => Some(t),
            _ => None,
        }
    }
}

impl Paint for GradRadial {
    fn at(&self, point: Point) -> LinColor {
        let offset = match self.offset(point) {
            None => return LinColor::new(0.0, 0.0, 0.0, 0.0),
            Some(offset) => offset,
        };
        let color = self.stops.at(self.spread.at(offset));
        if self.linear_colors {
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

    fn to_json(&self) -> Result<serde_json::Value, SvgParserError> {
        let ser: GradRadialSerialize = self.clone().into();
        Ok(serde_json::to_value(ser)?)
    }
}

impl Serialize for GradRadial {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        GradRadialSerialize::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GradRadial {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(GradRadialSerialize::deserialize(deserializer)?.into())
    }
}

/// Intermediate type for radial gradient (de)serialization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GradRadialSerialize {
    #[serde(rename = "type")]
    grad_type: String,
    #[serde(default, skip_serializing_if = "crate::utils::is_default")]
    units: Units,
    #[serde(default, skip_serializing_if = "crate::utils::is_default")]
    linear_colors: bool,
    #[serde(default, skip_serializing_if = "crate::utils::is_default")]
    spread: GradSpread,
    #[serde(with = "crate::utils::serde_from_str")]
    tr: Transform,
    center: Point,
    radius: Scalar,
    #[serde(default, skip_serializing_if = "crate::utils::is_default")]
    fcenter: Option<Point>,
    #[serde(default, skip_serializing_if = "crate::utils::is_default")]
    fradius: Scalar,
    stops: GradStops,
}

impl From<GradRadial> for GradRadialSerialize {
    fn from(mut grad: GradRadial) -> Self {
        if !grad.linear_colors {
            grad.stops.convert_to_linear();
        }
        Self {
            grad_type: GradRadial::GRAD_TYPE.into(),
            units: grad.units,
            linear_colors: grad.linear_colors,
            spread: grad.spread,
            tr: grad.tr,
            center: grad.center,
            radius: grad.radius,
            fcenter: (grad.center != grad.fcenter).then(|| grad.fcenter),
            fradius: grad.fradius,
            stops: grad.stops,
        }
    }
}

impl From<GradRadialSerialize> for GradRadial {
    fn from(value: GradRadialSerialize) -> Self {
        Self::new(
            value.stops,
            value.units,
            value.linear_colors,
            value.spread,
            value.tr,
            value.center,
            value.radius,
            value.fcenter.unwrap_or(value.center),
            value.fradius,
        )
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

    #[test]
    fn test_grad_stops() -> Result<(), Box<dyn std::error::Error>> {
        let stops = GradStops::new(vec![
            GradStop::new(0.0, LinColor::new(1.0, 0.0, 0.0, 1.0)),
            GradStop::new(0.5, LinColor::new(0.0, 1.0, 0.0, 1.0)),
            GradStop::new(1.0, LinColor::new(0.0, 0.0, 1.0, 1.0)),
        ]);
        assert_eq!(stops.at(-1.0), LinColor::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(stops.at(0.25), LinColor::new(0.5, 0.5, 0.0, 1.0));
        assert_eq!(stops.at(0.75), LinColor::new(0.0, 0.5, 0.5, 1.0));
        assert_eq!(stops.at(2.0), LinColor::new(0.0, 0.0, 1.0, 1.0));

        Ok(())
    }

    #[test]
    fn test_radial_grad() -> Result<(), Box<dyn std::error::Error>> {
        let fcenter = Point::new(0.25, 0.0);
        let center = Point::new(0.5, 0.0);
        let grad = GradRadial::new(
            vec![],
            Units::BoundingBox,
            true,
            GradSpread::Pad,
            Transform::identity(),
            center,
            0.5,
            fcenter,
            0.1,
        );
        assert!(grad.offset(fcenter).unwrap() < 0.0);
        assert_approx_eq!(grad.offset(Point::new(0.675, 0.0)).unwrap(), 0.5);
        assert_approx_eq!(grad.offset(Point::new(1.0, 0.0)).unwrap(), 1.0);

        let grad_str = serde_json::to_string(&grad)?;
        assert_eq!(grad, serde_json::from_str(grad_str.as_str())?);

        Ok(())
    }

    #[test]
    fn test_ling_grad() -> Result<(), Box<dyn std::error::Error>> {
        let c0 = "#89155180".parse()?;
        let c1 = "#ff272d80".parse()?;
        let c2 = "#ff272d00".parse()?;
        let grad = GradLinear::new(
            vec![
                GradStop::new(0.0, c0),
                GradStop::new(0.5, c1),
                GradStop::new(1.0, c2),
            ],
            Units::UserSpaceOnUse,
            true,
            GradSpread::default(),
            Transform::identity(),
            (0.0, 0.0),
            (1.0, 1.0),
        );

        assert_eq!(grad.at(Point::new(-0.5, -0.5)), c0);
        assert_eq!(grad.at(Point::new(0.0, 0.0)), c0);
        assert_eq!(grad.at(Point::new(1.0, 0.0)), c1);
        assert_eq!(grad.at(Point::new(0.0, 1.0)), c1);
        assert_eq!(grad.at(Point::new(1.0, 1.0)), c2);
        assert_eq!(grad.at(Point::new(1.5, 1.5)), c2);

        let grad_str = serde_json::to_string(&grad)?;
        assert_eq!(grad, serde_json::from_str(grad_str.as_str())?);

        Ok(())
    }
}
