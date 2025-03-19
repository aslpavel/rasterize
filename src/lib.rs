//! Simple 2D library that support SVG path parsing/generation/manipulation and rasterization.
//!
//! ## Main features:
//!  - SVG path parsing and generation
//!  - Anti-aliased rendering
//!  - Path offsetting [`Path::stroke`]
//!  - Linear and Radial gradients with [`GradLinear`] and [`GradRadial`]
//!  - Serde integration if `serde` feature is set (enabled by default)
//!
//! ## Overview
//! **Main types are:**
//! - [`Path`] - Represents the same concept as an [SVG path](https://www.w3.org/TR/SVG11/paths.html),
//!   the easiest way to construct it is with [`Path::builder`] or it can be parsed from SVG path with [`str::parse`].
//!   Path can be stroked with [`Path::stroke`] to generated new path that represents an outline.
//! - [`Scene`] - Represents an image that has not been rendered yet, multiple scenes
//!   can be composed to construct more complex scene. This is probably the simplest
//!   way to render something useful. See `examples/simple.rs` for a simple example.
//!   It can also be (de)serialized see `data/firefox.scene` for an example.
//! - [`Paint`] - Color/Gradient that can be used to fill a path.
//! - [`Image`] - 2D matrix that can hold and image and used as a target for rendering.
//!   Image can also be written into a file with [`Image::write_bmp`] or to PNG with
//!   `Image::write_png` if `png` feature is enabled.
#![deny(warnings)]
#![allow(clippy::excessive_precision)]

mod color;
mod curve;
mod ellipse;
mod geometry;
mod grad;
mod image;
mod path;
mod rasterize;
mod scene;
pub mod simd;
mod svg;
pub mod utils;

pub use crate::rasterize::{
    ActiveEdgeIter, ActiveEdgeRasterizer, ArcPaint, Paint, Pixel, Rasterizer,
    SignedDifferenceRasterizer, Size, Units,
};
#[cfg(feature = "serde")]
pub use color::RGBADeserializer;
pub use color::{Color, ColorError, LinColor, RGBA, SVG_COLORS, linear_to_srgb, srgb_to_linear};
pub use curve::{
    Cubic, Curve, CurveExtremities, CurveFlattenIter, CurveRoots, Line, Quad, Segment,
};
pub use ellipse::EllipArc;
pub use geometry::{
    Align, BBox, EPSILON, EPSILON_SQRT, PI, Point, Scalar, ScalarFormat, ScalarFormatter, Transform,
};
pub use grad::{GradLinear, GradRadial, GradSpread, GradStop, GradStops};
pub use image::{
    Image, ImageIter, ImageMut, ImageMutIter, ImageMutRef, ImageOwned, ImageRef, ImageWriteFormat,
    Shape,
};
pub use path::{
    DEFAULT_FLATNESS, FillRule, LineCap, LineJoin, Path, PathBuilder, StrokeStyle, SubPath,
};
pub use scene::{Layer, Scene};
pub use svg::{SvgParserError, SvgPathCmd, SvgPathParser};
