//! Simple example that randers a squirrel with a frame around it
use rasterize::*;
use std::{fs::File, io::BufWriter, sync::Arc};

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    // Create path using builder
    let squirrel_path: Arc<Path> = Path::builder()
        .move_to((12.0, 1.0))
        .cubic_to((9.79, 1.0), (8.0, 2.31), (8.0, 3.92))
        .cubic_to((8.0, 5.86), (8.5, 6.95), (8.0, 10.0))
        .cubic_to((8.0, 5.5), (5.23, 3.66), (4.0, 3.66))
        .cubic_to((4.05, 3.16), (3.52, 3.0), (3.52, 3.0))
        .cubic_to((3.52, 3.0), (3.3, 3.11), (3.22, 3.34))
        .cubic_to((2.95, 3.03), (2.66, 3.07), (2.66, 3.07))
        .line_to((2.53, 3.65))
        .cubic_to((2.53, 3.65), (0.7, 4.29), (0.68, 6.87))
        .cubic_to((0.88, 7.2), (2.21, 7.47), (3.15, 7.3))
        .cubic_to((4.04, 7.35), (3.82, 8.09), (3.62, 8.29))
        .cubic_to((2.78, 9.13), (2.0, 8.0), (1.0, 8.0))
        .cubic_to((0.0, 8.0), (0.0, 9.0), (1.0, 9.0))
        .cubic_to((2.0, 9.0), (2.0, 10.0), (4.0, 10.0))
        .cubic_to((0.91, 11.2), (4.0, 14.0), (4.0, 14.0))
        .line_to((3.0, 14.0))
        .cubic_to((2.0, 14.0), (2.0, 15.0), (2.0, 15.0))
        .line_to((8.0, 15.0))
        .cubic_to((11.0, 15.0), (13.0, 14.0), (13.0, 11.53))
        .cubic_to((13.0, 10.68), (12.57, 9.74), (12.0, 9.0))
        .cubic_to((10.89, 7.54), (12.23, 6.32), (13.0, 7.0))
        .cubic_to((13.77, 7.68), (16.0, 8.0), (16.0, 5.0))
        .cubic_to((16.0, 2.79), (14.21, 1.0), (12.0, 1.0))
        .close()
        .move_to((2.5, 6.0))
        .cubic_to((2.22, 6.0), (2.0, 5.78), (2.0, 5.5))
        .cubic_to((2.0, 5.22), (2.22, 5.0), (2.5, 5.0))
        .cubic_to((2.78, 5.0), (3.0, 5.22), (3.0, 5.5))
        .cubic_to((3.0, 5.78), (2.78, 6.0), (2.5, 6.0))
        .close()
        .build()
        .into();
    // Squirrels binding box
    let squirrel_bbox = squirrel_path
        .bbox(Transform::identity())
        .ok_or("Empty path")?;

    // Construct squirrel scene stroke and fill
    let squirrel_scene = Scene::group(vec![
        Scene::fill(
            squirrel_path.clone(),
            Arc::new("#d65d0e".parse::<LinColor>()?),
            FillRule::default(),
        ),
        Scene::stroke(
            squirrel_path,
            Arc::new("#af3a03".parse::<LinColor>()?),
            StrokeStyle {
                width: 0.3,
                line_join: LineJoin::default(),
                line_cap: LineCap::Round,
            },
        ),
    ])
    // fit it into 200x200 box with 10 pixel margin
    .transform(Transform::fit_bbox(
        squirrel_bbox,
        BBox::new((10.0, 10.0), (190.0, 190.0)),
        Align::Mid,
    ));

    // Construct frame around squirrel
    let frame_path: Arc<_> = Path::builder()
        .move_to((5.0, 5.0))
        .rbox((190.0, 190.0), (15.0, 15.0))
        .build()
        .into();
    // fill frame with gradient
    let frame_scene = Scene::group(vec![
        Scene::fill(
            frame_path.clone(),
            Arc::new("#fbf1c7".parse::<LinColor>()?),
            FillRule::NonZero,
        ),
        Scene::stroke(
            frame_path,
            Arc::new("#98971a".parse::<LinColor>()?),
            StrokeStyle {
                width: 2.0,
                line_join: LineJoin::default(),
                line_cap: LineCap::Round,
            },
        ),
    ]);

    // Construct full scene
    let scene =
        Scene::group(vec![frame_scene, squirrel_scene]).transform(Transform::new_scale(2.0, 2.0));

    // Render image
    let image = scene.render(
        &ActiveEdgeRasterizer::default(),
        Transform::identity(),
        Some(BBox::new((0.0, 0.0), (400.0, 400.0))),
        None,
    );

    // Write file
    //
    // Image pixels can also be iterated with
    // `for pixel in image.iter().map(Color::to_rgb) { ... }`
    static IMAGE_NAME: &str = "squirrel.bmp";
    let mut output = BufWriter::new(File::create(IMAGE_NAME)?);
    image.write_bmp(&mut output)?;
    println!("Image written to {IMAGE_NAME}");

    Ok(())
}
