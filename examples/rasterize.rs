//! Very simple tool that accepts SVG path as an input and produces rasterized image
#![deny(warnings)]

use rasterize::*;
use std::{
    env,
    fs::File,
    io::{BufWriter, Read},
    sync::Arc,
};
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

type Error = Box<dyn std::error::Error>;

#[derive(Debug, Clone, Copy)]
enum RasterizerType {
    ActiveEdge,
    SignedDifference,
}

#[derive(Debug)]
struct Args {
    input_file: String,
    output_file: String,
    outline: bool,
    width: Option<usize>,
    stroke: Option<Scalar>,
    flatness: Scalar,
    rasterizer: RasterizerType,
    tr: Transform,
    fg: Option<LinColor>,
    bg: Option<LinColor>,
    bbox: Option<BBox>,
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
            outline: false,
            width: None,
            stroke: None,
            flatness: DEFAULT_FLATNESS,
            rasterizer: RasterizerType::SignedDifference,
            tr: Transform::identity(),
            fg: None,
            bg: None,
            bbox: None,
        };
        let mut positional = 0;
        let mut args = env::args();
        let cmd = args.next().unwrap();
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
                "-b" => {
                    let bbox = args.next().ok_or("-b requires argument")?;
                    result.bbox = Some(bbox.parse()?);
                }
                "-t" => {
                    result.tr = args.next().ok_or("-t requires argument")?.parse()?;
                }
                "-s" => {
                    let stroke = args.next().ok_or("-s requres argument")?;
                    result.stroke = Some(stroke.parse()?);
                }
                "-o" => {
                    result.outline = true;
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
                "    {} [-w <width>] [-b <bbox>] [-s <stroke>] [-f <flatness>] [-o] [-a] [-fg <color>] [-bg <color>] <file.path> <out.bmp>",
                cmd
            );
            eprintln!("\nARGS:");
            eprintln!("    -w <width>         width in pixels of the output image");
            eprintln!("    -b <bbox>          custom bounding box");
            eprintln!("    -t <transform>     apply transform");
            eprintln!("    -s <stroke_width>  stroke path before rendering");
            eprintln!("    -o                 show outline with control points instead of filling");
            eprintln!(
                "    -a                 use active-edge instead of signed-difference rasterizer"
            );
            eprintln!("    -fg <color>        foreground color");
            eprintln!("    -bg <color>        background color");
            eprintln!(
                "    -f <flatness>      flatness used by rasterizer (defualt: {})",
                DEFAULT_FLATNESS
            );
            eprintln!("    <file.path>        file containing SVG path ('-' means stdin)");
            eprintln!("    <out.bmp>          image rendered in the BMP format ('-' means stdout)");
            std::process::exit(1);
        }
        Ok(result)
    }
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
    Ok(tracing::debug_span!("[parse]").in_scope(|| contents.parse())?)
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

    let path = match args.stroke {
        None => path_load(args.input_file)?,
        Some(stroke_width) => {
            let path = path_load(args.input_file)?;
            let stroke_style = StrokeStyle {
                width: stroke_width,
                line_join: LineJoin::Round,
                line_cap: LineCap::Round,
            };
            tracing::debug_span!("[stroke]").in_scope(|| path.stroke(stroke_style))
        }
    };
    let path = Arc::new(path);
    tracing::debug!("[path:segments_count] {}", path.segments_count());

    // transform if needed
    let tr = match args.width {
        Some(width) if width > 2 => {
            let src_bbox = match args.bbox {
                Some(bbox) => bbox.transform(args.tr),
                None => path.bbox(args.tr).ok_or("path is empty")?,
            };
            let width = width as Scalar;
            let height = src_bbox.height() * width / src_bbox.width();
            let dst_bbox = BBox::new(Point::new(1.0, 1.0), Point::new(width - 1.0, height - 1.0));
            Transform::fit_bbox(src_bbox, dst_bbox, Align::Mid) * args.tr
        }
        _ => args.tr,
    };

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
    let bbox = match args.bbox {
        Some(bbox) => bbox.transform(tr),
        None => scene
            .bbox(Transform::identity())
            .ok_or("nothing to render")?,
    };
    let bbox = BBox::new((bbox.x().round(), bbox.y().round()), bbox.max());
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
            image.write_bmp(&mut image_file)?;
        } else {
            image.write_bmp(std::io::stdout())?;
        }
    }

    Ok(())
}
