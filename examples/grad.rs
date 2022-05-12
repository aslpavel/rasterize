use rasterize::*;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fill_rule = FillRule::default();
    let rasterizer = ActiveEdgeRasterizer::default();
    // let rasterizer = SignedDifferenceRasterizer::default();

    /*
    let lin = GradLinear::new(
        vec![
            GradStop::new(0.0, LinColor::new(1.0, 0.0, 0.0, 1.0)),
            GradStop::new(0.5, LinColor::new(0.0, 1.0, 0.0, 1.0)),
            GradStop::new(1.0, LinColor::new(0.0, 0.0, 1.0, 1.0)),
        ],
        Units::BoundingBox,
        false,
        GradSpread::Pad,
        Transform::identity(),
        (0.0, 0.0),
        (1.0, 1.0),
    );
    let path = Arc::new(Path::read_svg_path(File::open("./data/squirrel.path")?)?);
    let scene = Scene::group(vec![Scene::fill(path.clone(), Arc::new(paint), fill_rule)]);
     */

    /*
    let path = PathBuilder::new()
        .move_to((10.0, 10.0))
        .rbox((100.0, 100.0), (15.0, 15.0))
        .build();
    let rad = GradRadial::new(
        vec![
            GradStop::new(0.0, LinColor::new(1.0, 0.0, 0.0, 1.0)),
            GradStop::new(1.0, LinColor::new(0.0, 0.0, 1.0, 1.0)),
        ],
        Units::BoundingBox,
        false,
        GradSpread::Pad,
        Transform::identity(),
        (0.5, 0.5),
        0.5,
        (0.25, 0.25),
        0.1,
    );
    let scene = Scene::fill(Arc::new(path), Arc::new(rad), fill_rule);
     */

    let rad = GradRadial::new(
        vec![
            GradStop::new(0.0, "#ffbd4f".parse()?),
            GradStop::new(0.5, "#ff0000".parse()?),
            GradStop::new(0.7, "#ff00ff".parse()?),
            GradStop::new(1.0, "#000000".parse()?),
        ],
        Units::UserSpaceOnUse,
        false,
        GradSpread::Reflect,
        Transform::new_skew(deg2rad(30.0), 0.0)
            .pre_translate(-30.0, -20.0)
            .pre_rotate(deg2rad(20.0)),
        (50.0, 50.0),
        40.0,
        (50.0, 50.0),
        0.0,
    );
    let scene = Scene::fill(
        Arc::new("M10,10 70,10 70,70 10,70z".parse()?),
        Arc::new(rad),
        fill_rule,
    );

    let img = scene.render(
        &rasterizer,
        Transform::new_rotate(deg2rad(30.0)).pre_translate(10.0, 10.0),
        Some(BBox::new((0.0, 0.0), (80.0, 80.0))),
        None,
    );
    img.write_bmp(std::io::stdout())?;

    Ok(())
}

fn deg2rad(rad: Scalar) -> Scalar {
    PI * rad / 180.0
}
