This is a fully functional 2D rasterizer and SVG path parser.

### Usage example
There is a very simple binary in examples folder that can be used to render SVG path
```
$ cargo run --release --example rasterize -- paths/squirrel.path -w 512 -o squirrel.ppm
```
This will produce
![squirrel](paths/squirrel.png)
