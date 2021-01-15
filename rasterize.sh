#!/usr/bin/env bash
CARGO=$(dirname $(realpath "${BASH_SOURCE[0]}"))/Cargo.toml
export RUST_LOG=debug
exec cargo run --manifest-path="$CARGO" --release --example rasterize -- "${@}"
