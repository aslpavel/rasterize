//! Simple 2D library that support SVG path parsing/generation/manipulation and rasterization.
//!
//! Main features:
//!  - SVG path parsing and generation
//!  - Anit-aliased rendering
//!  - Path offsetting
//!
#![deny(warnings)]

mod color;
mod curve;
mod ellipse;
mod geometry;
mod image;
mod path;
mod rasterize;
mod svg;
mod utils;

pub use crate::rasterize::{
    ActiveEdgeIter, ActiveEdgeRasterizer, Pixel, Rasterizer, SignedDifferenceRasterizer, Size,
};
pub use color::{Color, ColorU8, LinColor};
pub use curve::{
    Cubic, Curve, CurveExtremities, CurveFlattenIter, CurveRoots, Line, Quad, Segment,
};
pub use ellipse::EllipArc;
pub use geometry::{scalar_fmt, Align, BBox, Point, Scalar, Transform, EPSILON, EPSILON_SQRT, PI};
pub use image::{
    Image, ImageIter, ImageMut, ImageMutIter, ImageMutRef, ImageOwned, ImageRef, Shape,
};
pub use path::{
    FillRule, LineCap, LineJoin, Path, PathBuilder, StrokeStyle, SubPath, DEFAULT_FLATNESS,
};
pub use svg::{SVGPathCmd, SVGPathParser, SVGPathParserError};
use utils::{clamp, cubic_solve, quadratic_solve, ArrayIter, M3x3, M4x4};
