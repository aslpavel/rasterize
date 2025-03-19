//! Render scene from JSON serialized representation
#![deny(warnings)]
use rasterize::*;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read},
};
use tracing::debug_span;
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

type Error = Box<dyn std::error::Error>;

#[derive(Default)]
struct Args {
    input_file: String,
    output_file: String,
    tr: Option<Transform>,
    view_box: Option<BBox>,
    bg: Option<LinColor>,
    size: Size,
    output_format: ImageWriteFormat,
}

impl Args {
    fn parse() -> Result<Self, Error> {
        let mut result = Self::default();
        let mut args = std::env::args();
        let mut positional = 0;
        let _cmd = args.next().unwrap_or_else(|| "scene".to_string());
        while let Some(arg) = args.next() {
            match arg.as_ref() {
                "-h" => {
                    result.size.height = args.next().ok_or("-h requires argument")?.parse()?;
                }
                "-w" => {
                    result.size.width = args.next().ok_or("-w requires argument")?.parse()?;
                }
                "-v" => {
                    let view_box = args.next().ok_or("-v requires argument")?;
                    result.view_box.replace(view_box.parse()?);
                }
                "-t" => {
                    result
                        .tr
                        .replace(args.next().ok_or("-t requires argument")?.parse()?);
                }
                "-f" => {
                    result.output_format = args.next().ok_or("-of requries argument")?.parse()?;
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
            eprintln!("    scene [-h <height>] [-w <width>] [-v <view_box>] [-t <transform>]",);
            eprintln!("          [-f <format>] [-bg <color>] <scene> <output>",);
            eprintln!();
            eprintln!("ARGS:");
            eprintln!("    -h <height>     height in pixels of the output image");
            eprintln!("    -w <width>      width in pixels of the output image");
            eprintln!("    -v <view_box>   view box");
            eprintln!("    -t <transform>  apply transform");
            eprintln!("    -f <format>     output file format (bmp, png, rgba)");
            eprintln!("    -bg <color>     background color");
            eprintln!("    <scene>         file containing JSON scene ('-' means stdin)");
            eprintln!("    <output>        rendered scene ('-' means stdout)");
            eprintln!("\nVERSION: {}", env!("CARGO_PKG_VERSION"));
            std::process::exit(1);
        }

        Ok(result)
    }
}

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse()?;

    // scene
    let scene: Scene = {
        let file: &mut dyn Read = match args.input_file.as_str() {
            "-" => &mut std::io::stdin(),
            input_file => &mut File::open(input_file)?,
        };
        serde_json::from_reader(BufReader::new(file))?
    };
    let scene = match args.tr {
        None => scene,
        Some(tr) => scene.transform(tr),
    };

    // render transform
    let Some(view_box) = args.view_box.or_else(|| scene.bbox(Transform::identity())) else {
        return Err("nothing to render".into());
    };
    let (size, tr) = Transform::fit_size(view_box, args.size, Align::Mid);
    let bbox = BBox::new((0.0, 0.0), (size.width as Scalar, size.height as Scalar));

    // render
    let image = debug_span!("[render]")
        .in_scope(|| scene.render(&ActiveEdgeRasterizer::default(), tr, Some(bbox), args.bg));

    // save
    let save = debug_span!("[save]");
    {
        let _ = save.enter();
        if args.output_file != "-" {
            let mut image_file = BufWriter::new(File::create(args.output_file)?);
            image.write(args.output_format, &mut image_file)?;
        } else {
            image.write(args.output_format, std::io::stdout())?;
        }
    }

    Ok(())
}
