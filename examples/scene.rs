//! Render scene from JSON serialized representation
#![deny(warnings)]
use rasterize::*;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
};
type Error = Box<dyn std::error::Error>;

#[derive(Default)]
struct Args {
    input_file: String,
    output_file: String,
    tr: Transform,
    bg: Option<LinColor>,
    width: Option<usize>,
}

impl Args {
    fn parse() -> Result<Self, Error> {
        let mut result = Self::default();
        let mut args = std::env::args();
        let mut positional = 0;
        let cmd = args.next().unwrap_or_else(|| "scene".to_string());
        while let Some(arg) = args.next() {
            match arg.as_ref() {
                "-h" => {
                    positional = 0;
                    break;
                }
                "-w" => {
                    let width = args.next().ok_or("-w requires argument")?;
                    result.width = Some(width.parse()?);
                }
                "-t" => {
                    result.tr = args.next().ok_or("-t requires argument")?.parse()?;
                }
                "-bg" => {
                    let bg = args
                        .next()
                        .ok_or("-bg requres color #rrggbb(aa) argument")?
                        .parse()?;
                    result.bg = Some(bg);
                }
                _ => {
                    positional += 1;
                    match positional {
                        1 => result.input_file = arg,
                        2 => result.output_file = arg,
                        _ => return Err("unexpected positional argment".into()),
                    }
                }
            }
        }

        if positional < 2 {
            eprintln!("Render scene from JSON serialized representation");
            eprintln!();
            eprintln!("USAGE:");
            eprintln!(
                "    {} [-w <width>] [-bg <color>] [-t <transform>] <scene> <output>",
                cmd
            );
            eprintln!();
            eprintln!("ARGS:");
            eprintln!("    -w <width>      width in pixels of the output image");
            eprintln!("    -t <transform>  apply transform");
            eprintln!("    -bg <color>     background color");
            eprintln!("    <scene>         file containing JSON scene ('-' means stdin)");
            eprintln!("    <output>        rendered scene ('-' means stdout)");
            std::process::exit(1);
        }

        Ok(result)
    }
}

fn main() -> Result<(), Error> {
    let args = Args::parse()?;

    let scene: Scene = if args.input_file != "-" {
        let input = BufReader::new(File::open(args.input_file)?);
        serde_json::from_reader(input)?
    } else {
        serde_json::from_reader(BufReader::new(std::io::stdin()))?
    };

    let tr = match args.width {
        Some(width) if width > 2 => {
            let src_bbox = scene.bbox(args.tr).ok_or("scene is empty")?;
            let width = width as Scalar;
            let height = src_bbox.height() * width / src_bbox.width();
            let dst_bbox = BBox::new(Point::new(1.0, 1.0), Point::new(width - 1.0, height - 1.0));
            Transform::fit_bbox(src_bbox, dst_bbox, Align::Mid) * args.tr
        }
        _ => args.tr,
    };

    let image = scene.render(&ActiveEdgeRasterizer::default(), tr, None, args.bg);

    if args.output_file != "-" {
        let mut image_file = BufWriter::new(File::create(args.output_file)?);
        image.write_bmp(&mut image_file)?;
    } else {
        image.write_bmp(std::io::stdout())?;
    }

    Ok(())
}
