//! Very simple tool that accepts SVG path as an input and produces rasterized image
#![deny(warnings)]

use rasterize::*;
use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    str::FromStr,
    sync::Arc,
};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

type Error = Box<dyn std::error::Error>;

#[derive(Debug, Clone, Copy)]
enum RasterizerType {
    ActiveEdge,
    SignedDifference,
}

#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Bmp,
    Rgba,
    #[cfg(feature = "png")]
    Png,
}

impl OutputFormat {
    fn write(self, image: &Layer<LinColor>, out: impl Write) -> Result<(), Error> {
        match self {
            OutputFormat::Bmp => image.write_bmp(out)?,
            #[cfg(feature = "png")]
            OutputFormat::Png => image.write_png(out)?,
            OutputFormat::Rgba => image.write_rgba(out)?,
        }
        Ok(())
    }
}

impl FromStr for OutputFormat {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bmp" => Ok(OutputFormat::Bmp),
            "rgba" => Ok(OutputFormat::Rgba),
            #[cfg(feature = "png")]
            "png" => Ok(OutputFormat::Png),
            #[cfg(not(feature = "png"))]
            "png" => Err("png feature is disabled".into()),
            _ => Err(format!("Invalid output format: {s}").into()),
        }
    }
}

#[derive(Debug)]
struct Args {
    input_file: String,
    output_file: String,
    output_format: OutputFormat,
    outline: bool,
    size: Size,
    stroke: Option<Scalar>,
    flatness: Scalar,
    rasterizer: RasterizerType,
    tr: Option<Transform>,
    fg: Option<LinColor>,
    bg: Option<LinColor>,
    view_box: Option<BBox>,
}

impl Args {
    fn get_rasterizer(&self) -> Box<dyn Rasterizer> {
        use RasterizerType::*;
        match self.rasterizer {
            SignedDifference => Box::new(SignedDifferenceRasterizer::new(self.flatness)),
            ActiveEdge => Box::new(ActiveEdgeRasterizer::new(self.flatness)),
        }
    }

    fn parse() -> Result<Args, Error> {
        let mut result = Args {
            input_file: String::new(),
            output_file: String::new(),
            output_format: OutputFormat::Bmp,
            outline: false,
            size: Size {
                height: 0,
                width: 0,
            },
            stroke: None,
            flatness: DEFAULT_FLATNESS,
            rasterizer: RasterizerType::SignedDifference,
            tr: None,
            fg: None,
            bg: None,
            view_box: None,
        };
        let mut positional = 0;
        let mut args = env::args();
        let _cmd = args.next().unwrap();
        while let Some(arg) = args.next() {
            match arg.as_ref() {
                "-h" => {
                    result.size.height = args.next().ok_or("-h requires argument")?.parse()?;
                }
                "-w" => {
                    result.size.width = args.next().ok_or("-w requires argument")?.parse()?;
                }
                "-b" => {
                    let view_box = args.next().ok_or("-b requires argument")?;
                    result.view_box.replace(view_box.parse()?);
                }
                "-t" => {
                    let tr = args.next().ok_or("-t requires argument")?.parse()?;
                    result.tr.replace(tr);
                }
                "-s" => {
                    let stroke = args.next().ok_or("-s requres argument")?;
                    result.stroke.replace(stroke.parse()?);
                }
                "-o" => {
                    result.outline = true;
                }
                "-of" => {
                    result.output_format = args.next().ok_or("-of requries argument")?.parse()?;
                }
                "-a" => {
                    result.rasterizer = RasterizerType::ActiveEdge;
                }
                "-f" => {
                    let flatness: Scalar = args.next().ok_or("-f requres argument")?.parse()?;
                    if flatness < EPSILON {
                        return Err("flatness is too small".into());
                    }
                    result.flatness = flatness;
                }
                "-fg" => {
                    let fg = args
                        .next()
                        .ok_or("-fg requres color #rrggbb(aa) argument")?
                        .parse()?;
                    result.fg.replace(fg);
                }
                "-bg" => {
                    let bg: LinColor = args
                        .next()
                        .ok_or("-bg requres color #rrggbb(aa) argument")?
                        .parse()?;
                    result.bg.replace(bg);
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
            eprintln!(
                "Very simple tool that accepts SVG path as an input and produces rasterized image"
            );
            eprintln!("\nUSAGE:");
            eprintln!(
                "    rasterize [-h <height>] [-w <width>] [-b <bbox>] [-t <transform>] [-s <stroke>]",
            );
            eprintln!(
                "              [-f <flatness>] [-o] [-of <format>] [-a] [-fg <color>] [-bg <color>]",
            );
            eprintln!("              <input_file> <output_file>");
            eprintln!("\nARGS:");
            eprintln!("    -h <height>        height in pixels of the output image");
            eprintln!("    -w <width>         width in pixels of the output image");
            eprintln!("    -b <view_box>      view box");
            eprintln!("    -t <transform>     apply transform");
            eprintln!("    -s <stroke_width>  stroke path before rendering");
            eprintln!("    -o                 show outline with control points instead of filling");
            eprintln!("    -of <format>       output file format (bmp, png, rgba)");
            eprintln!(
                "    -a                 use active-edge instead of signed-difference rasterizer"
            );
            eprintln!("    -fg <color>        foreground color");
            eprintln!("    -bg <color>        background color");
            eprintln!(
                "    -f <flatness>      flatness used by rasterizer (defualt: {})",
                DEFAULT_FLATNESS
            );
            eprintln!("    <input_file>       file containing SVG path ('-' means stdin)");
            eprintln!("    <output_file>      image rendered in the BMP format ('-' means stdout)");
            std::process::exit(1);
        }
        Ok(result)
    }
}

/// Convert path to the outline with control points.
fn outline(path: &Path, tr: Transform) -> Scene {
    let mut path = path.clone();
    path.transform(tr);
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
    for subpath in path.subpaths() {
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
                &Path::builder()
                    .move_to(segment.start())
                    .circle(control_radius)
                    .build(),
            );
            output.extend(&control.build().stroke(control_style));
        }
        if (subpath.start() - subpath.end()).length() > control_radius {
            output.extend(
                &Path::builder()
                    .move_to(subpath.end())
                    .circle(control_radius)
                    .build(),
            );
        }
    }
    Scene::fill(
        Arc::new(output),
        Arc::new(LinColor::new(0.0, 0.0, 0.0, 1.0)),
        FillRule::default(),
    )
}

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::CLOSE)
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse()?;
    let rasterizer = args.get_rasterizer();

    // load path
    let mut path = {
        let file: &mut dyn Read = match args.input_file.as_str() {
            "-" => &mut std::io::stdin(),
            input_file => &mut File::open(input_file)?,
        };
        tracing::debug_span!("[parse]").in_scope(|| Path::read_svg_path(BufReader::new(file)))?
    };

    // transform
    if let Some(tr) = args.tr {
        path.transform(tr);
    }

    // stroke
    if let Some(stroke_width) = args.stroke {
        let stroke_style = StrokeStyle {
            width: stroke_width,
            line_join: LineJoin::Round,
            line_cap: LineCap::Round,
        };
        path = tracing::debug_span!("[stroke]").in_scope(|| path.stroke(stroke_style));
    }

    // allocate path
    let path = Arc::new(path);
    tracing::debug!("[path:segments_count] {}", path.segments_count());

    // render transform
    let Some(view_box) = args.view_box.or_else(|| path.bbox(Transform::identity())) else {
        return Err("nothing to render".into());
    };
    tracing::debug!(?view_box, "[view box]");
    let (size, tr) = Transform::fit_size(view_box, args.size, Align::Mid);
    let bbox = BBox::new((0.0, 0.0), (size.width as Scalar, size.height as Scalar));

    // scene
    let mut group = Vec::new();
    let fg = match args.fg {
        Some(fg) => fg,
        None => {
            if args.outline {
                LinColor::new(0.0, 0.0, 0.0, 0.6)
            } else {
                LinColor::new(0.0, 0.0, 0.0, 1.0)
            }
        }
    };
    group.push(Scene::fill(path.clone(), Arc::new(fg), FillRule::default()).transform(tr));
    if args.outline {
        group.push(outline(&path, tr));
    }
    let scene = Scene::group(group);

    // add background or checkerboard
    let (scene, bg) = match args.bg {
        None => {
            let scene = Scene::group(vec![
                Scene::fill(
                    Arc::new(Path::builder().checkerboard(bbox, 16.0).build()),
                    Arc::new("#d0d0d0".parse::<LinColor>()?),
                    FillRule::EvenOdd,
                ),
                scene,
            ]);
            (scene, "#f0f0f0".parse()?)
        }
        Some(bg) => (scene, bg),
    };

    let image = tracing::debug_span!("[render]", rasterizer = rasterizer.name())
        .in_scope(|| scene.render(&rasterizer, Transform::identity(), Some(bbox), Some(bg)));

    // save
    let save = tracing::debug_span!("[save]");
    {
        let _ = save.enter();
        if args.output_file != "-" {
            let mut image_file = BufWriter::new(File::create(args.output_file)?);
            args.output_format.write(&image, &mut image_file)?;
        } else {
            args.output_format.write(&image, std::io::stdout())?;
        }
    }

    Ok(())
}
