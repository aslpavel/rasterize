#!/usr/bin/env bash
cd $(dirname "${BASH_SOURCE[0]}")
exec cargo run --release --example rasterize -- "${@}"
