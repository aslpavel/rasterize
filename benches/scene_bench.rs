#![deny(warnings)]
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rasterize::{BBox, FillRule, LinColor, Path, Point, RGBA, Scalar, Scene, Size, Transform};
use std::sync::Arc;

fn many_cirles_benchmark(c: &mut Criterion) {
    let mut rnd = Rnd::new();

    // scene
    let size = Size {
        height: 1024,
        width: 1024,
    };
    let fsize = Point::new(size.height as Scalar, size.width as Scalar);
    let count = 1024;

    let mut group = Vec::new();
    for _ in 0..count {
        let circle = Path::builder()
            .move_to(rnd.point() * fsize)
            .circle(rnd.uniform() * 10.0 + 30.0)
            .build();
        group.push(Scene::fill(
            circle.into(),
            Arc::new(rnd.color()),
            FillRule::default(),
        ))
    }
    let scene = Scene::group(group);
    let view = Some(BBox::new((0.0, 0.0), fsize));

    let mut group = c.benchmark_group("many-circles");
    group.throughput(Throughput::Elements(count));

    // signed difference
    let rasterizer = &rasterize::SignedDifferenceRasterizer::default();
    group.bench_function("signed-difference", |b| {
        b.iter_with_large_drop(|| scene.render(rasterizer, Transform::identity(), view, None))
    });

    // signed difference
    let rasterizer = &rasterize::ActiveEdgeRasterizer::default();
    group.bench_function("active-edge", |b| {
        b.iter_with_large_drop(|| scene.render(rasterizer, Transform::identity(), view, None))
    });
}

criterion_group!(scene, many_cirles_benchmark);
criterion_main!(scene);

/// Very basic random number generator
#[derive(Default)]
pub struct Rnd {
    state: u32,
}

impl Rnd {
    /// Create new random number generator with seed `0`
    pub fn new() -> Self {
        Self::default()
    }

    /// Create new random number generator with provided `seed` value
    pub fn with_seed(seed: u32) -> Self {
        Self { state: seed }
    }

    fn step(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(214_013).wrapping_add(2_531_011) & 0x7fffffff;
        self.state >> 16
    }

    /// Sample `u32` from uniform distributes
    pub fn uniform_u32(&mut self) -> u32 {
        (self.step() & 0xffff) << 16 | (self.step() & 0xffff)
    }

    /// Sample `u64` from uniform distributes
    pub fn uniform_u64(&mut self) -> u64 {
        ((self.uniform_u32() as u64) << 32) | (self.uniform_u32() as u64)
    }

    /// Sample f64 from `Uniform([0, 1])`
    pub fn uniform(&mut self) -> f64 {
        let bpr_recip: f64 = (2.0f64).powi(-53);
        (self.uniform_u64() >> 10) as f64 * bpr_recip
    }

    /// Sample form normal/gauss distribution with `mu = 0`, `sigma = 1`
    pub fn gauss(&mut self) -> f64 {
        const PI_2: f64 = 2.0 * std::f64::consts::PI;
        let x = self.uniform();
        let y = self.uniform();
        (PI_2 * x).cos() * (-2.0 * (1.0 - y).ln()).sqrt()
    }

    /// Generate random color
    pub fn color(&mut self) -> LinColor {
        RGBA::new(
            (self.uniform_u32() % 256) as u8,
            (self.uniform_u32() % 256) as u8,
            (self.uniform_u32() % 256) as u8,
            255,
        )
        .into()
    }

    /// Generate random `Point(U([0, 1]), U([0, 1]))`
    pub fn point(&mut self) -> Point {
        Point::new(self.uniform(), self.uniform())
    }
}
