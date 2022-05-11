/// Render scene from JSON serialized representation
use rasterize::*;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
};
type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().collect();
    let (input, output) = match args.as_slice() {
        [_, input, output] => (input, output),
        [cmd, ..] => {
            eprintln!("Render scene from JSON serialized representation");
            eprintln!("USAGE:");
            eprintln!("    {} <scene> <output>", cmd);
            std::process::exit(1);
        }
        _ => unreachable!(),
    };

    let scene: Scene = serde_json::from_reader(BufReader::new(File::open(input)?))?;
    let image = scene.render_pipeline(
        &ActiveEdgeRasterizer::default(),
        Transform::identity(),
        None,
        None,
    );

    if output != "-" {
        let mut image_file = BufWriter::new(File::create(output)?);
        image.write_bmp(&mut image_file)?;
    } else {
        image.write_bmp(std::io::stdout())?;
    }

    Ok(())
}
