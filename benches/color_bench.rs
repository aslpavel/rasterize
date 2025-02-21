use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use rasterize::{
    linear_to_srgb,
    simd::{f32x4, l2s},
};

fn linear_to_srgb_bench(c: &mut Criterion) {
    let v0 = [0.001, 0.1, 0.2, 0.7];
    let v1: f32x4 = v0.into();

    let mut group = c.benchmark_group("linear_to_srgb");
    group.throughput(Throughput::Elements(1));
    group.bench_function("basic", |b| b.iter(|| l2s_basic(black_box(v0))));
    group.bench_function("simd", |b| b.iter(|| l2s(black_box(v1))));
}

fn l2s_basic(c: [f32; 4]) -> [f32; 4] {
    [
        linear_to_srgb(c[0]),
        linear_to_srgb(c[1]),
        linear_to_srgb(c[2]),
        linear_to_srgb(c[3]),
    ]
}

criterion_group!(color, linear_to_srgb_bench);
criterion_main!(color);
