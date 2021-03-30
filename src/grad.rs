use crate::{LinColor, Point, Scalar, Transform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GradSpread {
    Pad,
    Repeat,
    Reflect,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GradStop {
    position: Scalar,
    color: LinColor,
}

#[derive(Debug, Clone)]
struct GradStops {
    stops: Vec<GradStop>,
}

impl GradStops {
    fn at(t: Scalar) -> LinColor;
}

struct GradLinear {
    stops: GradStops,
    start: Point,
    end: Point,
    spread: GradSpread,
    transform: Transform,
}

impl Paint for GradLinear {}
