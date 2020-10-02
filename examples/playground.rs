#![allow(unused_imports, dead_code)]
use env_logger::Env;
use rasterize::{
    surf_to_ppm, timeit, Cubic, Curve, FillRule, Line, LineCap, LineJoin, Path, Quad, StrokeStyle,
    SubPath, Transform,
};
use std::{
    env,
    fs::File,
    io::{BufWriter, Read},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::from_env(Env::default().default_filter_or("debug")).init();
    let path = Path::load(File::open("./paths/rust.path")?)?;
    // let path: Path = "M 0 0 L 5 8 L 9 0z".parse()?;

    let stroke_style = StrokeStyle {
        width: 2.0,
        line_join: LineJoin::Round,
        line_cap: LineCap::Round,
    };
    let stroke = path.stroke(stroke_style);
    println!("{}", stroke.to_svg_path());

    Ok(())
}
