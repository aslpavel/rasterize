#![deny(warnings)]
use rasterize::*;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = Scene::fill(
        Arc::new("M10,10 h80 v80 h-80 Z".parse()?),
        Arc::new(GradLinear::new(
            vec![
                GradStop::new(0.01, "#89155180".parse()?),
                GradStop::new(0.48, "#ff272d80".parse()?),
                GradStop::new(1.0, "#ff272d00".parse()?),
            ],
            Units::BoundingBox,
            true,
            GradSpread::default(),
            Transform::identity(),
            (0.0, 0.0),
            (0.8, 0.0),
        )),
        FillRule::default(),
    );

    let rasterizer = ActiveEdgeRasterizer::default();
    let img = scene.render(
        &rasterizer,
        Transform::identity().pre_scale(0.32, 0.32),
        Some(BBox::new((0.0, 0.0), (32.0, 32.0))),
        Some("#b16286".parse()?),
    );
    img.write_bmp(std::io::stdout())?;

    Ok(())
}
