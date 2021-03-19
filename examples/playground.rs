#![deny(warnings)]
use rasterize::*;
use std::{fs::File, sync::Arc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Arc::new(Path::read_svg_path(File::open("./paths/squirrel.path")?)?);
    let fill_rule = FillRule::default();
    let tr = Transform::default();
    let bbox = path.bbox(tr).expect("path is empty");

    // let rasterizer = ActiveEdgeRasterizer::default();
    let rasterizer = SignedDifferenceRasterizer::default();

    let scene = Scene::group(vec![
        Scene::fill(
            path.clone(),
            Arc::new(LinColor::new(1.0, 0.0, 0.0, 1.0)),
            fill_rule,
        ),
        Scene::fill(
            path.clone(),
            Arc::new(LinColor::new(0.0, 1.0, 0.0, 1.0)),
            fill_rule,
        )
        .transform(Transform::default().translate(bbox.width() / 3.0, 0.0)),
        Scene::fill(path, Arc::new(LinColor::new(0.0, 0.0, 1.0, 1.0)), fill_rule)
            .transform(Transform::default().translate(bbox.width() * 2.0 / 3.0, 0.0)),
    ]);

    /*
    let mut img = ImageOwned::new_default(Size {
        height: bbox.height() as usize + 2,
        width: (bbox.width() * 2.0) as usize + 2,
    });

    path.fill(
        &rasterizer,
        tr,
        fill_rule,
        LinColor::new(1.0, 0.0, 0.0, 1.0),
        img.as_mut(),
    );

    path.fill(
        &rasterizer,
        tr.translate(bbox.width() / 3.0, 0.0),
        fill_rule,
        LinColor::new(0.0, 1.0, 0.0, 1.0),
        img.as_mut(),
    );

    path.fill(
        &rasterizer,
        tr.translate(bbox.width() * 2.0 / 3.0, 0.0),
        fill_rule,
        LinColor::new(0.0, 0.0, 1.0, 1.0),
        img.as_mut(),
    );
    */

    let img = scene.render(&rasterizer, Transform::default(), bbox);
    img.write_bmp(std::io::stdout())?;

    Ok(())
}
