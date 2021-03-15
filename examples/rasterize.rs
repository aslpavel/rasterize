//! Very simple tool that accepts SVG path as an input and produces rasterized image
#![deny(warnings)]

use env_logger::Env;
use rasterize::{
    ActiveEdgeRasterizer, Align, BBox, Curve, FillRule, Image, ImageOwned, LineCap, LineJoin, Path,
    Point, Rasterizer, Scalar, Segment, SignedDifferenceRasterizer, StrokeStyle, Transform,
};
use std::{
    env, fmt,
    fs::File,
    io::{BufWriter, Read},
};

/// Add debug log message with time taken to execute provided function
fn timeit<F: FnOnce() -> R, R>(msg: impl AsRef<str>, f: F) -> R {
    let start = std::time::Instant::now();
    let result = f();
    log::debug!("{} {:?}", msg.as_ref(), start.elapsed());
    result
}

type Error = Box<dyn std::error::Error>;

#[derive(Debug)]
struct ArgsError(String);

impl ArgsError {
    fn new(err: impl Into<String>) -> Self {
        Self(err.into())
    }
}

impl fmt::Display for ArgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ArgsError {}

struct Args {
    input_file: String,
    output_file: String,
    outline: bool,
    width: Option<usize>,
    stroke: Option<Scalar>,
    rasterizer: Box<dyn Rasterizer>,
}

fn parse_args() -> Result<Args, Error> {
    let mut result = Args {
        input_file: String::new(),
        output_file: String::new(),
        outline: false,
        width: None,
        stroke: None,
        rasterizer: Box::new(SignedDifferenceRasterizer::default()),
    };
    let mut postional = 0;
    let mut args = env::args();
    let cmd = args.next().unwrap();
    while let Some(arg) = args.next() {
        match arg.as_ref() {
            "-w" => {
                let width = args
                    .next()
                    .ok_or_else(|| ArgsError::new("-w requires argument"))?;
                result.width = Some(width.parse()?);
            }
            "-s" => {
                let stroke = args
                    .next()
                    .ok_or_else(|| ArgsError::new("-s requres argument"))?;
                result.stroke = Some(stroke.parse()?);
            }
            "-o" => {
                result.outline = true;
            }
            "-a" => {
                result.rasterizer = Box::new(ActiveEdgeRasterizer::default());
            }
            _ => {
                postional += 1;
                match postional {
                    1 => result.input_file = arg,
                    2 => result.output_file = arg,
                    _ => return Err(ArgsError::new("unexpected positional argment").into()),
                }
            }
        }
    }
    if postional < 2 {
        eprintln!(
            "Very simple tool that accepts SVG path as an input and produces rasterized image"
        );
        eprintln!("\nUSAGE:");
        eprintln!(
            "    {} [-w <width>] [-s <stroke>] [-o] [-a] <file.path> <out.bmp>",
            cmd
        );
        eprintln!("\nARGS:");
        eprintln!("    -w <width>         width in pixels of the output image");
        eprintln!("    -s <stroke_width>  stroke path before rendering");
        eprintln!("    -o                 show outline with control points instead of filling");
        eprintln!("    -a                 use active-edge instead of signed-difference rasterizer");
        eprintln!("    <file.path>        file containing SVG path ('-' means stdin)");
        eprintln!("    <out.bmp>          image rendered in the BMP format ('-' means stdout)");
        std::process::exit(1);
    }
    Ok(result)
}

/// Load path for the file
fn path_load(path: String) -> Result<Path, Error> {
    let mut contents = String::new();
    if path != "-" {
        let mut file = File::open(path)?;
        file.read_to_string(&mut contents)?;
    } else {
        std::io::stdin().read_to_string(&mut contents)?;
    }
    Ok(timeit("[parse]", || contents.parse())?)
}

/// Convert path to the outline with control points.
fn outline(path: &Path) -> Path {
    let stroke_style = StrokeStyle {
        width: 2.0,
        line_join: LineJoin::Round,
        line_cap: LineCap::Round,
    };
    let control_style = StrokeStyle {
        width: 1.0,
        ..stroke_style
    };
    let control_radius = 3.0;
    let mut output = path.stroke(stroke_style);
    for subpath in path.subpaths().iter() {
        for segment in subpath.segments() {
            let mut control = Path::builder();
            match segment {
                Segment::Line(_) => {}
                Segment::Quad(quad) => {
                    let [p0, p1, p2] = quad.points();
                    control
                        .move_to(p0)
                        .line_to(p1)
                        .circle(control_radius)
                        .line_to(p2);
                }
                Segment::Cubic(cubic) => {
                    let [p0, p1, p2, p3] = cubic.points();
                    control
                        .move_to(p0)
                        .line_to(p1)
                        .circle(control_radius)
                        .move_to(p3)
                        .line_to(p2)
                        .circle(control_radius);
                }
            };
            output.extend(
                Path::builder()
                    .move_to(segment.start())
                    .circle(control_radius)
                    .build(),
            );
            output.extend(control.build().stroke(control_style));
        }
        if (subpath.start() - subpath.end()).length() > control_radius {
            output.extend(
                Path::builder()
                    .move_to(subpath.end())
                    .circle(control_radius)
                    .build(),
            );
        }
    }
    output
}

fn main() -> Result<(), Error> {
    env_logger::from_env(Env::default().default_filter_or("debug")).init();
    let args = parse_args()?;

    let mut path = path_load(args.input_file)?;

    // resize if needed
    let tr = match args.width {
        Some(width) if width > 2 => {
            let src_bbox = path
                .bbox(Transform::default())
                .ok_or_else(|| ArgsError::new("path is empty"))?;
            let width = width as Scalar;
            let height = src_bbox.height() * width / src_bbox.width();
            let dst_bbox = BBox::new(Point::new(1.0, 1.0), Point::new(width - 1.0, height - 1.0));
            Transform::fit_bbox(src_bbox, dst_bbox, Align::Mid)
        }
        _ => Transform::default(),
    };

    // stroke
    log::info!("[path::segments_count] {}", path.segments_count());
    if let Some(stroke) = args.stroke {
        path = timeit("[stroke]", || {
            path.stroke(StrokeStyle {
                width: stroke,
                line_join: LineJoin::Round,
                line_cap: LineCap::Round,
            })
        });
        log::info!("[stroke::segments_count] {}", path.segments_count());
    }

    // convert to outline with control points
    if args.outline {
        path = outline(&path);
    }

    // rasterize path
    let rasterizer = args.rasterizer;
    let (size, tr, _) = path.size(tr).expect("path is empty");
    let img = timeit("[allocate]", || ImageOwned::new_default(size));
    let mask = timeit(format!("[rasterize:{}]", rasterizer.name()), || {
        path.mask(rasterizer, tr, FillRule::NonZero, img)
    });
    log::info!(
        "[dimension] width: {} height: {}",
        mask.width(),
        mask.height()
    );

    // save
    if args.output_file != "-" {
        let mut image = BufWriter::new(File::create(args.output_file)?);
        timeit("[save:bmp]", || mask.write_bmp(&mut image))?;
    } else {
        timeit("[save:bmp]", || mask.write_bmp(std::io::stdout()))?;
    }

    Ok(())
}
