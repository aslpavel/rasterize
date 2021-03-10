//! Simple 2D library that support SVG path parsing/generation/manipulation and rasterization.
//!
//! Main features:
//!  - SVG path parsing and generation
//!  - Anit-aliased rendering
//!  - Path offsetting
//!
#![deny(warnings)]

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

use std::io::Write;

/// Save image as PPM stream
pub fn img_to_ppm<I, W>(img: I, mut w: W) -> Result<(), std::io::Error>
where
    I: Image,
    I::Pixel: Color,
    W: Write,
{
    write!(w, "P6 {} {} 255 ", img.width(), img.height())?;
    for color in img.iter() {
        w.write_all(&color.to_rgb())?;
    }
    Ok(())
}

/// Save image as PNG stream
#[cfg(feature = "png")]
pub fn img_to_png<I, W>(img: I, w: W) -> Result<(), png::EncodingError>
where
    I: Image,
    I::Pixel: Color,
    W: Write,
{
    let mut encoder = png::Encoder::new(w, img.width() as u32, img.height() as u32);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let mut stream_writer = writer.stream_writer();
    for color in img.iter() {
        stream_writer.write_all(&color.to_rgba())?;
    }
    stream_writer.flush()?;
    Ok(())
}

fn linear_to_srgb(value: Scalar) -> Scalar {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

pub trait Color {
    fn to_rgb(&self) -> [u8; 3];
    fn to_rgba(&self) -> [u8; 4];
}

impl Color for Scalar {
    fn to_rgb(&self) -> [u8; 3] {
        let color = (linear_to_srgb(1.0 - *self) * 255.0).round() as u8;
        [color; 3]
    }

    fn to_rgba(&self) -> [u8; 4] {
        // let color = (clamp(1.0 - *self, 0.0, 1.0) * 255.0).round() as u8;
        let color = (linear_to_srgb(1.0 - *self) * 255.0).round() as u8;
        [color, color, color, 255]
    }
}
