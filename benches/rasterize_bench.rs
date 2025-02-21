#![deny(warnings)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rasterize::*;
use std::{
    fs::File,
    io::{Cursor, Read},
    time::Duration,
};

const SQUIRREL_FILE: &str = "data/squirrel.path";
const MATERIAL_FILE: &str = "data/material-big.path";

fn rasterizers() -> impl Iterator<Item = Box<dyn Rasterizer>> {
    let r0: Box<dyn Rasterizer> = Box::new(SignedDifferenceRasterizer::default());
    let r1: Box<dyn Rasterizer> = Box::new(ActiveEdgeRasterizer::default());
    Some(r0).into_iter().chain(Some(r1))
}

fn curve_benchmark(c: &mut Criterion) {
    let cubic = Cubic::new((158.0, 70.0), (210.0, 250.0), (25.0, 190.0), (219.0, 89.0));
    let length = cubic.length(0.0, 1.0) / 2.0;
    let mut group = c.benchmark_group("cubic");
    group
        .throughput(Throughput::Elements(1))
        .bench_function("extremities", |b| b.iter(|| black_box(cubic).extremities()))
        .bench_function("bbox", |b| b.iter(|| black_box(cubic).bbox(None)))
        .bench_function("length", |b| b.iter(|| black_box(cubic).length(0.0, 1.0)))
        .bench_function("from length", |b| {
            b.iter(|| cubic.param_at_length(black_box(length), None))
        });
    group.finish();
}

fn stroke_benchmark(c: &mut Criterion) {
    let mut file = File::open(SQUIRREL_FILE).expect("failed to open path");
    let path = Path::read_svg_path(&mut file).expect("failed to load path");
    let (size, tr, _) = path.size(Transform::identity()).expect("path is empty");
    let mut img = ImageOwned::new_default(size);
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
        group
            .bench_with_input(
                BenchmarkId::new("mask stroked", rasterizer.name()),
                &rasterizer,
                |b, r| {
                    b.iter(|| {
                        img.clear();
                        stroke.mask(r, tr, FillRule::EvenOdd, &mut img);
                    })
                },
            )
            .bench_with_input(
                BenchmarkId::new("mask", rasterizer.name()),
                &rasterizer,
                |b, r| {
                    b.iter(|| {
                        img.clear();
                        path.mask(r, tr, FillRule::EvenOdd, &mut img);
                    });
                },
            );
    }
    group.finish()
}

fn large_path_benchmark(c: &mut Criterion) {
    let mut path_str = String::new();
    let mut file = File::open(MATERIAL_FILE).expect("failed to open a path");
    file.read_to_string(&mut path_str)
        .expect("failed to read path");
    let path: Path = path_str.parse().unwrap();
    let (size, tr, _) = path.size(Transform::identity()).expect("path is empty");
    let mut img = ImageOwned::new_default(size);

    let mut group = c.benchmark_group("material-big");
    group
        .throughput(Throughput::Elements(path.segments_count() as u64))
        .bench_function("parse-only", |b| {
            b.iter(|| SvgPathParser::new(Cursor::new(path_str.as_str())).count())
        })
        .bench_function("parse", |b| {
            b.iter_with_large_drop(|| path_str.parse::<Path>())
        })
        .bench_function("flatten", |b| {
            b.iter(|| path.flatten(tr, DEFAULT_FLATNESS, true).count())
        })
        .bench_function("bbox", |b| b.iter(|| path.bbox(tr)));
    for rasterizer in rasterizers() {
        let id = BenchmarkId::new("mask", rasterizer.name());
        group.bench_with_input(id, &rasterizer, |b, r| {
            b.iter(|| {
                img.clear();
                path.mask(r, tr, FillRule::EvenOdd, &mut img);
            })
        });
    }
    group.finish()
}

criterion_group!(
    name = rasterize;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::new(1, 0));
    targets = curve_benchmark, large_path_benchmark, stroke_benchmark
);
criterion_main!(rasterize);
