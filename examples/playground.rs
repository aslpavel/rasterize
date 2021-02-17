#![deny(warnings)]
use rasterize::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
