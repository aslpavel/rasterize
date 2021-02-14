#![deny(warnings)]

use std::{fmt, io::Write, iter::FromIterator, ops::Mul};
pub mod render;
pub use render::{
    Align, BBox, Cubic, Curve, Error, FillRule, Line, LineCap, LineJoin, Path, Point, Quad, Scalar,
    Segment, StrokeStyle, SubPath, Transform,
};
pub use surface::{Surface, SurfaceMut, SurfaceOwned};

/// Restrict value to a certain interval
#[inline]
pub fn clamp<T>(val: T, min: T, max: T) -> T
where
    T: PartialOrd,
{
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

/// Add debug log message with time taken to execute provided function
pub fn timeit<F: FnOnce() -> R, R>(msg: &str, f: F) -> R {
    let start = std::time::Instant::now();
    let result = f();
    log::debug!("{} {:?}", msg, start.elapsed());
    result
}

/// Save surface as PPM stream
pub fn surf_to_ppm<S, W>(surf: S, mut w: W) -> Result<(), std::io::Error>
where
    S: Surface,
    S::Item: Color,
    W: Write,
{
    write!(w, "P6 {} {} 255 ", surf.width(), surf.height())?;
    for color in surf.iter() {
        w.write_all(&color.to_rgb())?;
    }
    Ok(())
}

/// Save surface as PNG stream
#[cfg(feature = "png")]
pub fn surf_to_png<S, W>(surf: S, w: W) -> Result<(), png::EncodingError>
where
    S: Surface,
    S::Item: Color,
    W: Write,
{
    let mut encoder = png::Encoder::new(w, surf.width() as u32, surf.height() as u32);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let mut stream_writer = writer.stream_writer();
    for color in surf.iter() {
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

/// Abstraction over slices used by `ArrayIter`
pub trait Array {
    type Item;
    fn new() -> Self;
    fn size(&self) -> usize;
    fn at(&self, index: usize) -> Option<&Self::Item>;
    fn take(&mut self, index: usize) -> Option<Self::Item>;
    fn put(&mut self, index: usize, value: Self::Item) -> Option<Self::Item>;
}

macro_rules! impl_array(
    ($($size:expr),+) => {
        $(
            impl<T: Copy> Array for [Option<T>; $size] {
                type Item = T;
                fn new() -> Self {
                    [None; $size]
                }
                fn size(&self) -> usize { $size }
                fn at(&self, index: usize) -> Option<&Self::Item> {
                    self.get(index).and_then(|item| item.as_ref())
                }
                fn take(&mut self, index: usize) -> Option<Self::Item> {
                    self[index].take()
                }
                fn put(&mut self, index: usize, value: Self::Item) -> Option<Self::Item> {
                    self[index].replace(value)
                }
            }
        )+
    }
);

impl_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

/// Fixed sized iterator
///
/// This is similar to a smallvec but it never allocates and just panics if you try to fit
/// more data than expected.
#[derive(Clone, Copy)]
pub struct ArrayIter<A> {
    size: usize,
    consumed: usize,
    array: A,
}

impl<A> fmt::Debug for ArrayIter<A>
where
    A: Array,
    A::Item: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = fmt.debug_list();
        for index in self.consumed..self.size {
            self.array.at(index).map(|item| list.entry(item));
        }
        list.finish()?;
        Ok(())
    }
}

impl<A: Array> Default for ArrayIter<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Array> ArrayIter<A> {
    pub fn new() -> Self {
        Self {
            consumed: 0,
            size: 0,
            array: A::new(),
        }
    }

    /// Push new element to the end of the iterator
    pub fn push(&mut self, item: A::Item) {
        self.array.put(self.size, item);
        self.size += 1;
    }

    /// Check if array iterator is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of uncosumed elements
    pub fn len(&self) -> usize {
        self.size - self.consumed
    }
}

impl<A: Array> Extend<A::Item> for ArrayIter<A> {
    fn extend<T: IntoIterator<Item = A::Item>>(&mut self, iter: T) {
        for item in iter.into_iter() {
            self.push(item);
        }
    }
}

impl<A> FromIterator<A::Item> for ArrayIter<A>
where
    A: Array,
    A::Item: Copy,
{
    fn from_iter<T: IntoIterator<Item = A::Item>>(iter: T) -> Self {
        let mut array = ArrayIter::<A>::new();
        for item in iter.into_iter() {
            array.push(item);
        }
        array
    }
}

impl<A: Array> Iterator for ArrayIter<A> {
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed < self.size {
            let item = self.array.take(self.consumed);
            self.consumed += 1;
            item
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.size - self.consumed;
        (size, Some(size))
    }
}

#[derive(Debug, Clone, Copy)]
struct M3([Scalar; 9]);

impl Mul<M3> for M3 {
    type Output = M3;

    fn mul(self, other: Self) -> Self::Output {
        let M3(a) = self;
        let M3(b) = other;
        let mut out = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    out[k + 3 * i] += a[j + 3 * i] * b[k + 3 * j];
                }
            }
        }
        M3(out)
    }
}

#[derive(Debug, Clone, Copy)]
struct M4([Scalar; 16]);

impl Mul<M4> for M4 {
    type Output = M4;

    fn mul(self, other: Self) -> Self::Output {
        let M4(a) = self;
        let M4(b) = other;
        let mut out = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    out[k + 4 * i] += a[j + 4 * i] * b[k + 4 * j];
                }
            }
        }
        M4(out)
    }
}
