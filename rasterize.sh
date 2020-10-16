#!/usr/bin/env bash
cd $(dirname "${BASH_SOURCE[0]}")
export RUST_LOG=debug
exec cargo run --release --example rasterize -- "${@}"
