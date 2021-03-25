#![deny(warnings)]
use rasterize::*;
use std::{fs::File, sync::Arc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Arc::new(Path::read_svg_path(File::open("./paths/squirrel.path")?)?);
    let fill_rule = FillRule::default();
    let tr = Transform::identity();
    let bbox = path.bbox(tr).expect("path is empty");

    let rasterizer = ActiveEdgeRasterizer::default();
    // let rasterizer = SignedDifferenceRasterizer::default();

    let scene = Scene::group(vec![
        Scene::fill(
            Arc::new(
                Path::builder()
                    .move_to((bbox.min() + bbox.max()) / 2.0)
                    .circle(bbox.width().min(bbox.height()) / 2.0)
                    .build(),
            ),
            Arc::new(LinColor::new(0.59375, 0.58984375, 0.1015625, 1.0)),
            fill_rule,
        ),
        Scene::fill(
            path.clone(),
            Arc::new(LinColor::new(1.0, 0.0, 0.0, 1.0)),
            fill_rule,
        ),
        Scene::group(vec![
            Scene::fill(
                path.clone(),
                Arc::new(LinColor::new(0.0, 1.0, 0.0, 1.0)),
                fill_rule,
            )
            .transform(Transform::new_translate(bbox.width() / 3.0, 0.0)),
            Scene::fill(path, Arc::new(LinColor::new(0.0, 0.0, 1.0, 1.0)), fill_rule)
                .transform(Transform::new_translate(bbox.width() * 2.0 / 3.0, 0.0)),
        ])
        .opacity(0.8),
    ])
    .clip(
        Arc::new(
            Path::builder()
                .move_to((bbox.min() + bbox.max()) / 2.0)
                .circle(bbox.width().min(bbox.height()) / 2.0)
                .build(),
        ),
        Units::default(),
        fill_rule,
    );

    let img = scene.render(&rasterizer, Transform::new_scale(1.0, 1.0), None, None);
    img.write_bmp(std::io::stdout())?;

    Ok(())
}
