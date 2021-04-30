#![deny(warnings)]
use rasterize::*;
use std::{fs::File, sync::Arc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Arc::new(Path::read_svg_path(File::open("./paths/squirrel.path")?)?);
    let fill_rule = FillRule::default();
    // let tr = Transform::identity();
    // let bbox = path.bbox(tr).expect("path is empty");

    let rasterizer = ActiveEdgeRasterizer::default();
    // let rasterizer = SignedDifferenceRasterizer::default();

    let scene = Scene::fill(
        path.clone(),
        Arc::new(LinColor::new(1.0, 0.0, 0.0, 1.0)),
        fill_rule,
    )
    .clip(
        Arc::new(Path::builder().move_to((0.5, 0.5)).circle(0.5).build()),
        Units::BoundingBox,
        fill_rule,
    );

    let img = scene.render(
        &rasterizer,
        Transform::new_scale(10.0, 10.0).pre_rotate(PI / 3.0),
        None,
        None,
    );
    img.write_bmp(std::io::stdout())?;

    Ok(())
}
