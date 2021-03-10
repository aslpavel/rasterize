#![deny(warnings)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rasterize::*;
use std::{
    fs::File,
    io::{Cursor, Read},
    time::Duration,
};

const SQUIRREL_FILE: &str = "paths/squirrel.path";
const MATERIAL_FILE: &str = "paths/material-big.path";

fn rasterizers() -> impl Iterator<Item = Box<dyn Rasterizer>> {
    let r0: Box<dyn Rasterizer> = Box::new(SignedDifferenceRasterizer::default());
    let r1: Box<dyn Rasterizer> = Box::new(ActiveEdgeRasterizer::default());
    Some(r0).into_iter().chain(Some(r1))
}

fn curve_benchmark(c: &mut Criterion) {
    let path = Cubic::new((157.0, 67.0), (35.0, 200.0), (220.0, 260.0), (175.0, 45.0));
    c.bench_function("cubic extremities", |b| {
        b.iter(|| black_box(path).extremities())
    });
}

fn stroke_benchmark(c: &mut Criterion) {
    let mut file = File::open(SQUIRREL_FILE).expect("failed to open path");
    let path = Path::load(&mut file).expect("failed to load path");
    let tr = Transform::default();
    let style = StrokeStyle {
        width: 1.0,
        line_join: LineJoin::Round,
        line_cap: LineCap::Round,
    };
    let stroke = path.stroke(style);

    let mut group = c.benchmark_group("squirrel");
    group.throughput(Throughput::Elements(path.segments_count() as u64));
    group.bench_function("stroke", |b| b.iter_with_large_drop(|| path.stroke(style)));
    for rasterizer in rasterizers() {
        let id = BenchmarkId::new("rasterize stroked", rasterizer.name());
        group.bench_with_input(id, &rasterizer, |b, r| {
            b.iter_with_large_drop(|| stroke.rasterize(r, tr, FillRule::EvenOdd))
        });
        let id = BenchmarkId::new("rasterize fill", rasterizer.name());
        group.bench_with_input(id, &rasterizer, |b, r| {
            b.iter_with_large_drop(|| path.rasterize(r, tr, FillRule::EvenOdd))
        });
    }
    group.finish()
}

fn large_path_benchmark(c: &mut Criterion) {
    let tr = Transform::default();
    let mut path_str = String::new();
    let mut file = File::open(MATERIAL_FILE).expect("failed to open a path");
    file.read_to_string(&mut path_str)
        .expect("failed to read path");
    let path: Path = path_str.parse().unwrap();
    let mut img = path.rasterize(SignedDifferenceRasterizer::default(), tr, FillRule::EvenOdd);

    let mut group = c.benchmark_group("material-big");
    group.throughput(Throughput::Elements(path.segments_count() as u64));
    group.bench_function("parse-only", |b| {
        b.iter(|| SVGPathParser::new(Cursor::new(path_str.as_str())).count())
    });
    group.bench_function("parse", |b| {
        b.iter_with_large_drop(|| path_str.parse::<Path>())
    });
    group.bench_function("flatten", |b| {
        b.iter(|| path.flatten(tr, DEFAULT_FLATNESS, true).count())
    });
    group.bench_function("bbox", |b| b.iter(|| path.bbox(tr)));
    for rasterizer in rasterizers() {
        let id = BenchmarkId::new("rasterize", rasterizer.name());
        group.bench_with_input(id, &rasterizer, |b, r| {
            b.iter_with_large_drop(|| path.rasterize(r, tr, FillRule::EvenOdd))
        });
        let id = BenchmarkId::new("rasterize to", rasterizer.name());
        group.bench_with_input(id, &rasterizer, |b, r| {
            b.iter(|| {
                img.clear();
                path.rasterize_to(r, tr, FillRule::EvenOdd, &mut img);
            })
        });
    }
    group.finish()
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::new(1, 0));
    targets = curve_benchmark, large_path_benchmark, stroke_benchmark
);
criterion_main!(benches);
