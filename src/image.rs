use crate::{Color, Size};
use std::{io::Write, sync::Arc};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    /// Width of the image
    pub width: usize,
    /// Height of the image
    pub height: usize,
    /// How many elements we need to skip to get to the next row.
    pub row_stride: usize,
    /// How many elements we need to skip to get to the next column.
    pub col_stride: usize,
}

impl Shape {
    #[inline]
    pub fn offset(&self, row: usize, col: usize) -> usize {
        row * self.row_stride + col * self.col_stride
    }

    #[inline]
    pub fn nth(&self, n: usize) -> Option<(usize, usize)> {
        if self.width == 0 {
            return None;
        }
        let row = n / self.width;
        let col = n - row * self.width;
        (row < self.height).then(move || (row, col))
    }

    pub fn size(&self) -> Size {
        Size {
            width: self.width,
            height: self.height,
        }
    }
}

pub trait Image {
    /// Pixel type
    type Pixel;

    /// Data containing image
    fn data(&self) -> &[Self::Pixel];

    /// Shape of the image
    fn shape(&self) -> Shape;

    /// Image width
    fn width(&self) -> usize {
        self.shape().width
    }

    /// Image height
    fn height(&self) -> usize {
        self.shape().height
    }

    fn get(&self, row: usize, col: usize) -> Option<&Self::Pixel> {
        let offset = self.shape().offset(row, col);
        self.data().get(offset)
    }

    fn as_ref(&self) -> ImageRef<'_, Self::Pixel> {
        ImageRef {
            shape: self.shape(),
            data: self.data(),
        }
    }

    /// Iterate over pixels
    fn iter(&self) -> ImageIter<'_, Self::Pixel> {
        ImageIter {
            index: 0,
            shape: self.shape(),
            data: self.data(),
        }
    }

    /// Write image in PPM format
    fn write_ppm<W>(&self, mut out: W) -> Result<(), std::io::Error>
    where
        W: Write,
        Self::Pixel: Color,
        Self: Sized,
    {
        write!(out, "P6 {} {} 255 ", self.width(), self.height())?;
        for color in self.iter() {
            out.write_all(&color.to_rgb())?;
        }
        Ok(())
    }

    /// Write image in BMP format
    fn write_bmp<W>(&self, mut out: W) -> Result<(), std::io::Error>
    where
        W: Write,
        Self::Pixel: Color,
        Self: Sized,
    {
        const BMP_HEADER_SIZE: u32 = 14;
        const DIB_HEADER_SIZE: u32 = 108;

        let data_offset = BMP_HEADER_SIZE + DIB_HEADER_SIZE;
        let data_size = (self.width() * self.height() * 4) as u32;

        // BMP File Header
        write!(out, "BM")?;
        out.write_all(&(data_size + data_offset).to_le_bytes())?; // file size
        out.write_all(&[0; 4])?; // reserved
        out.write_all(&data_offset.to_le_bytes())?; // image data offset

        // [BITMAPV4HEADER](https://docs.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapv4header)
        out.write_all(&DIB_HEADER_SIZE.to_le_bytes())?; // DIB header size
        out.write_all(&(self.width() as u32).to_le_bytes())?; // image width
        out.write_all(&(self.height() as u32).to_le_bytes())?; // image height
        out.write_all(&[0x01, 0x00])?; // planes
        out.write_all(&[0x20, 0x00])?; // 32 bits per pixel
        out.write_all(&[0x03, 0x00, 0x00, 0x00])?; // BI_BITFIELDS format
        out.write_all(&data_size.to_le_bytes())?; // image data size
        out.write_all(&[0x13, 0x0B, 0x00, 0x00])?; // horizontal print resolution (2835 = 72dpi)
        out.write_all(&[0x13, 0x0B, 0x00, 0x00])?; // vertical print resolution (2835 = 72dpi)
        out.write_all(&[0x00; 8])?; // palette colors are not used
        out.write_all(&[0xFF, 0x00, 0x00, 0x00])?; // blue mask
        out.write_all(&[0x00, 0xFF, 0x00, 0x00])?; // green mask
        out.write_all(&[0x00, 0x00, 0xFF, 0x00])?; // red mask
        out.write_all(&[0x00, 0x00, 0x00, 0xFF])?; // alpha mask
        out.write_all(&[0x42, 0x47, 0x52, 0x73])?; // sRGB color space
        out.write_all(&[0x00; 48])?; // not used

        let data = self.data();
        let shape = self.shape();
        for row in (0..shape.height).into_iter().rev() {
            for col in 0..shape.width {
                let color = &data[shape.offset(row, col)];
                out.write_all(&color.to_rgba())?;
            }
        }

        Ok(())
    }

    /// Write image in PNG format
    #[cfg(feature = "png")]
    fn write_png<W>(&self, out: W) -> Result<(), png::EncodingError>
    where
        W: Write,
        Self::Pixel: Color,
        Self: Sized,
    {
        let mut encoder = png::Encoder::new(out, self.width() as u32, self.height() as u32);
        encoder.set_color(png::ColorType::RGBA);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        let mut stream_writer = writer.stream_writer();
        for color in self.iter() {
            stream_writer.write_all(&color.to_rgba())?;
        }
        stream_writer.flush()?;
        Ok(())
    }
}

pub struct ImageIter<'a, P> {
    index: usize,
    shape: Shape,
    data: &'a [P],
}

impl<'a, P> ImageIter<'a, P> {
    pub fn position(&self) -> (usize, usize) {
        self.shape.nth(self.index).unwrap_or((self.shape.height, 0))
    }
}

impl<'a, P> Iterator for ImageIter<'a, P> {
    type Item = &'a P;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n + 1;
        let (row, col) = self.shape.nth(self.index - 1)?;
        self.data.get(self.shape.offset(row, col))
    }
}

pub trait ImageMut: Image {
    fn data_mut(&mut self) -> &mut [Self::Pixel];

    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Pixel> {
        let index = self.shape().offset(row, col);
        self.data_mut().get_mut(index)
    }

    fn as_mut(&mut self) -> ImageMutRef<'_, Self::Pixel> {
        ImageMutRef {
            shape: self.shape(),
            data: self.data_mut(),
        }
    }

    fn clear(&mut self)
    where
        Self::Pixel: Default,
    {
        let shape = self.shape();
        let data = self.data_mut();
        for row in 0..shape.height {
            for col in 0..shape.width {
                data[shape.offset(row, col)] = Default::default();
            }
        }
    }

    fn iter_mut(&mut self) -> ImageMutIter<'_, Self::Pixel> {
        ImageMutIter {
            index: 0,
            shape: self.shape(),
            data: self.data_mut(),
        }
    }
}

pub struct ImageMutIter<'a, P> {
    index: usize,
    shape: Shape,
    data: &'a mut [P],
}

impl<'a, P> ImageMutIter<'a, P> {
    pub fn position(&self) -> (usize, usize) {
        self.shape.nth(self.index).unwrap_or((self.shape.height, 0))
    }
}

impl<'a, P> Iterator for ImageMutIter<'a, P> {
    type Item = &'a mut P;

    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index += n + 1;
        let (row, col) = self.shape.nth(self.index - 1)?;
        let offset = self.shape.offset(row, col);

        if offset >= self.data.len() {
            None
        } else {
            // this is safe, iterator is always progressing and never
            // returns a mutable reference to the same location.
            let ptr = self.data.as_mut_ptr();
            let item = unsafe { &mut *ptr.add(offset) };
            Some(item)
        }
    }
}

#[derive(Clone)]
pub struct ImageOwned<P> {
    shape: Shape,
    data: Vec<P>,
}

impl<P> ImageOwned<P> {
    pub fn new(shape: Shape, data: Vec<P>) -> Self {
        Self { shape, data }
    }

    pub fn new_default(height: usize, width: usize) -> Self
    where
        P: Default,
    {
        Self::new_with(height, width, |_, _| Default::default())
    }

    pub fn new_with<F>(height: usize, width: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> P,
    {
        let mut data = Vec::with_capacity(height * width);
        for row in 0..height {
            for col in 0..width {
                data.push(f(row, col))
            }
        }
        Self {
            shape: Shape {
                width,
                height,
                row_stride: width,
                col_stride: 1,
            },
            data,
        }
    }

    pub fn to_vec(self) -> Vec<P> {
        self.data
    }
}

impl<P> Image for ImageOwned<P> {
    type Pixel = P;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Pixel] {
        &self.data
    }
}

impl<C> ImageMut for ImageOwned<C> {
    fn data_mut(&mut self) -> &mut [Self::Pixel] {
        &mut self.data
    }
}

#[derive(Clone)]
pub struct ImageRef<'a, P> {
    shape: Shape,
    data: &'a [P],
}

impl<'a, P> ImageRef<'a, P> {
    pub fn new(shape: Shape, data: &'a [P]) -> Self {
        Self { shape, data }
    }
}

impl<'a, P> Image for ImageRef<'a, P> {
    type Pixel = P;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Pixel] {
        self.data
    }
}

pub struct ImageMutRef<'a, C> {
    shape: Shape,
    data: &'a mut [C],
}

impl<'a, P> ImageMutRef<'a, P> {
    pub fn new(shape: Shape, data: &'a mut [P]) -> Self {
        Self { shape, data }
    }
}

impl<'a, P> Image for ImageMutRef<'a, P> {
    type Pixel = P;

    fn shape(&self) -> Shape {
        self.shape
    }

    fn data(&self) -> &[Self::Pixel] {
        self.data
    }
}

impl<'a, P> ImageMut for ImageMutRef<'a, P> {
    fn data_mut(&mut self) -> &mut [Self::Pixel] {
        self.data
    }
}

impl<'a, I> Image for &'a I
where
    I: Image + ?Sized,
{
    type Pixel = I::Pixel;

    fn shape(&self) -> Shape {
        (*self).shape()
    }

    fn data(&self) -> &[Self::Pixel] {
        (*self).data()
    }
}

impl<'a, I> Image for &'a mut I
where
    I: Image + ?Sized,
{
    type Pixel = I::Pixel;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Pixel] {
        (**self).data()
    }
}

impl<'a, I> ImageMut for &'a mut I
where
    I: ImageMut + ?Sized,
{
    fn data_mut(&mut self) -> &mut [Self::Pixel] {
        (**self).data_mut()
    }
}

impl<P> Image for Arc<dyn Image<Pixel = P>> {
    type Pixel = P;

    fn shape(&self) -> Shape {
        (**self).shape()
    }

    fn data(&self) -> &[Self::Pixel] {
        (**self).data()
    }
}
