# Rasterizer
![Build Status](https://github.com/aslpavel/rasterize/actions/workflows/rust.yml/badge.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Crate](https://img.shields.io/crates/v/rasterize.svg)](https://crates.io/crates/rasterize)
[![API Docs](https://docs.rs/rasterize/badge.svg)](https://docs.rs/rasterize)

This is a fully functional 2D rasterizer and SVG path parser.
### Features:
- parsing SVG path format
- rendering elliptic arc, lines, cubic/quadratic bezier curves
- curve offsetting
- linear and radial gradients

### Usage example
There is a very simple binary in examples folder that can be used to render SVG path
```
$ cargo run --release --example rasterize -- paths/squirrel.path -w 512 -o squirrel.bmp
```
This will produce:

![squirrel](https://raw.githubusercontent.com/aslpavel/rasterize/main/paths/squirrel.png)
