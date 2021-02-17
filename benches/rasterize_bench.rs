#![deny(warnings)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rasterize::{
    Cubic, Curve, FillRule, LineCap, LineJoin, Path, StrokeStyle, SurfaceMut, Transform,
    DEFAULT_FLATNESS,
};
use std::{fs::File, io::Read, time::Duration};

fn curve_benchmark(c: &mut Criterion) {
    let path = Cubic::new((157.0, 67.0), (35.0, 200.0), (220.0, 260.0), (175.0, 45.0));
    c.bench_function("cubic extremities", |b| {
        b.iter(|| black_box(path).extremities())
    });
}

fn stroke_benchmark(c: &mut Criterion) {
    let mut file = File::open("paths/squirrel.path").expect("failed to open path");
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
    group.bench_function("resterize stroked", |b| {
        b.iter_with_large_drop(|| stroke.rasterize(tr, FillRule::EvenOdd))
    });
    group.bench_function("resterize fill", |b| {
        b.iter_with_large_drop(|| path.rasterize(tr, FillRule::EvenOdd))
    });
    group.finish()
}

fn large_path_benchmark(c: &mut Criterion) {
    let tr = Transform::default();
    let mut path_str = String::new();
    let mut file = File::open("paths/material-big.path").expect("failed to open a path");
    file.read_to_string(&mut path_str)
        .expect("failed to read path");
    let path: Path = path_str.parse().unwrap();
    let mut surf = path.rasterize(tr, FillRule::EvenOdd);

    let mut group = c.benchmark_group("material-big");
    group.throughput(Throughput::Elements(path.segments_count() as u64));
    group.bench_function("parse", |b| {
        b.iter_with_large_drop(|| path_str.parse::<Path>())
    });
    group.bench_function("flatten", |b| {
        b.iter(|| path.flatten(tr, DEFAULT_FLATNESS, true).count())
    });
    group.bench_function("bbox", |b| b.iter(|| path.bbox(tr)));
    group.bench_function("rasterize", |b| {
        b.iter_with_large_drop(|| path.rasterize(tr, FillRule::EvenOdd))
    });
    group.bench_function("rasterize to", |b| {
        b.iter(|| {
            surf.clear();
            path.rasterize_to(tr, FillRule::EvenOdd, &mut surf);
        })
    });
    group.finish()
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::new(1, 0));
    targets = curve_benchmark, large_path_benchmark, stroke_benchmark
);
criterion_main!(benches);
