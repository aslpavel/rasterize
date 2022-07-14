use std::{sync::Arc, time::Instant};

use rasterize::{
    BBox, ColorU8, FillRule, Image, LinColor, Path, Point, Scalar, Scene, Size, Transform,
};

type Error = Box<dyn std::error::Error>;

pub struct Rnd {
    state: u32,
}

impl Default for Rnd {
    fn default() -> Self {
        Self::new()
    }
}

impl Rnd {
    /// Create new random number generator with seed `0`
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Create new random number generator with provided `seed` value
    pub fn with_seed(seed: u32) -> Self {
        Self { state: seed }
    }

    fn step(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(214_013).wrapping_add(2_531_011) & 0x7fffffff;
        self.state >> 16
    }

    /// Generate random u32 value
    pub fn next_u32(&mut self) -> u32 {
        (self.step() & 0xffff) << 16 | (self.step() & 0xffff)
    }

    /// Generate random u64 value
    pub fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }

    pub fn uniform(&mut self) -> f64 {
        let bpr_recip: f64 = (2.0f64).powi(-53);
        (self.next_u64() >> 10) as f64 * bpr_recip
    }

    pub fn gauss(&mut self) -> f64 {
        const PI_2: f64 = 2.0 * std::f64::consts::PI;
        let x = self.uniform();
        let y = self.uniform();
        (PI_2 * x).cos() * (-2.0 * (1.0 - y).ln()).sqrt()
    }

    pub fn next_color(&mut self) -> LinColor {
        ColorU8::new(
            (self.next_u32() % 256) as u8,
            (self.next_u32() % 256) as u8,
            (self.next_u32() % 256) as u8,
            255,
        )
        .into()
    }

    pub fn next_point(&mut self) -> Point {
        Point::new(self.uniform(), self.uniform())
    }
}

fn main() -> Result<(), Error> {
    let size = Size {
        height: 1024,
        width: 1024,
    };
    let fsize = Point::new(size.height as Scalar, size.width as Scalar);
    let count = 1024;
    let mut rnd = Rnd::new();

    let mut group = Vec::new();
    for _ in 0..count {
        let circle = Path::builder()
            .move_to(rnd.next_point() * fsize)
            .circle(rnd.uniform() * 10.0 + 30.0)
            .build();
        group.push(Scene::fill(
            circle.into(),
            Arc::new(rnd.next_color()),
            FillRule::default(),
        ))
    }
    let scene = Scene::group(group);
    let now = Instant::now();
    let rasterize = &rasterize::ActiveEdgeRasterizer::default();
    // let rasterize = &rasterize::SignedDifferenceRasterizer::default();
    let img = scene.render(
        rasterize,
        Transform::identity(),
        Some(BBox::new((0.0, 0.0), fsize)),
        Some("#ffffff".parse()?),
    );
    eprintln!("{:?}", now.elapsed());
    img.write_bmp(std::io::stdout())?;

    Ok(())
}
