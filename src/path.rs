use crate::{
    curve::line_offset, rasterize::Rasterizer, utils::clamp, BBox, Cubic, Curve, EllipArc,
    ImageMut, LinColor, Line, Paint, Point, Quad, Scalar, Segment, Size, SvgParserError,
    SvgPathParser, Transform, EPSILON,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    io::{Cursor, Read, Write},
    str::FromStr,
};

/// Default flatness used during rasterization.
/// Value of 0.05px gives good accuracy tradeoff.
pub const DEFAULT_FLATNESS: Scalar = 0.05;

/// The algorithm to use to determine the inside part of a shape, when filling it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FillRule {
    /// Fill area with non-zero winding number
    #[default]
    #[cfg_attr(feature = "serde", serde(rename = "nonzero"))]
    NonZero,
    /// Fill area with odd winding number
    #[cfg_attr(feature = "serde", serde(rename = "evenodd"))]
    EvenOdd,
}

impl FillRule {
    pub fn alpha_from_winding(&self, winding: Scalar) -> Scalar {
        match self {
            FillRule::EvenOdd => ((winding + 1.0).rem_euclid(2.0) - 1.0).abs(),
            FillRule::NonZero => {
                let value = winding.abs();
                if value >= 1.0 {
                    1.0
                } else if value < 1e-6 {
                    0.0
                } else {
                    value
                }
            }
        }
    }
}

impl FromStr for FillRule {
    type Err = SvgParserError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nonzero" => Ok(FillRule::NonZero),
            "evenodd" => Ok(FillRule::EvenOdd),
            _ => Err(SvgParserError::InvalidFillRule),
        }
    }
}

impl fmt::Display for FillRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FillRule::NonZero => "nonzero".fmt(f),
            FillRule::EvenOdd => "evenodd".fmt(f),
        }
    }
}

/// `LineJoin` defines the shape to be used at the corners of paths when they are stroked.
/// See [SVG specification](https://www.w3.org/TR/SVG2/painting.html#LineJoin) for more details.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "lowercase")
)]
pub enum LineJoin {
    /// Continue path segments with lines until they intersect. But only
    /// if `miter_length = stroke-width / sin(0.5 * eta)` is less than the miter argument.
    Miter(Scalar),
    /// Connect path segments with straight line.
    Bevel,
    /// Round corner is to be used to join path segments.
    /// The corner is a circular sector centered on the join point.
    Round,
}

impl Default for LineJoin {
    fn default() -> Self {
        Self::Miter(4.0)
    }
}

/// `LineCap` specifies the shape to be used at the end of open sub-paths when they are stroked.
/// See [SVG specification](https://www.w3.org/TR/SVG2/painting.html#LineCaps) for more details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "lowercase")
)]
pub enum LineCap {
    /// Connect path segments with straight line.
    Butt,
    /// Add half-square to the end of the segments
    Square,
    /// Add half-circle to the end of the segments
    Round,
}

impl Default for LineCap {
    fn default() -> Self {
        Self::Butt
    }
}

/// Style used to generate stroke
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StrokeStyle {
    /// Width of the stroke
    pub width: Scalar,
    /// How to join offset segments
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "crate::utils::is_default")
    )]
    pub line_join: LineJoin,
    /// How to join segments at the ends of the path
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "crate::utils::is_default")
    )]
    pub line_cap: LineCap,
}

/// Non-empty collections of segments where end of each segments coincides with the start of the next one.
#[derive(Clone, Copy, PartialEq)]
pub struct SubPath<'a> {
    /// List of segments representing SubPath
    segments: &'a [Segment],
    /// Whether SubPath contains an implicit line segment connecting start and the end of it.
    closed: bool,
}

impl<'a> fmt::Debug for SubPath<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for segment in self.segments.iter() {
            writeln!(f, "{:?}", segment)?;
        }
        if self.closed {
            writeln!(f, "Close")?;
        } else {
            writeln!(f, "End")?
        }
        Ok(())
    }
}

impl<'a> SubPath<'a> {
    pub fn new(segments: &'a [Segment], closed: bool) -> Option<Self> {
        if segments.is_empty() {
            None
        } else {
            Some(Self { segments, closed })
        }
    }

    /// Whether sub-path is closed or not
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// First segment in the sub-path
    pub fn first(&self) -> Segment {
        *self.segments.first().expect("SubPath is never emtpy")
    }

    /// Last segment in the sub-path
    pub fn last(&self) -> Segment {
        *self.segments.last().expect("SubPath is never empty")
    }

    pub fn flatten(
        &self,
        tr: Transform,
        flatness: Scalar,
        close: bool,
    ) -> impl Iterator<Item = Line> + '_ {
        let last = if self.closed || close {
            Some(Line::new(self.end(), self.start()).transform(tr))
        } else {
            None
        };
        self.segments
            .iter()
            .flat_map(move |segment| segment.flatten(tr, flatness))
            .chain(last)
    }

    /// Start point of the sub-path
    pub fn start(&self) -> Point {
        self.first().start()
    }

    /// End point of the sub-path
    pub fn end(&self) -> Point {
        self.last().end()
    }

    /// Bounding box of the sub-path
    pub fn bbox(&self, init: Option<BBox>, tr: Transform) -> BBox {
        self.segments
            .iter()
            .fold(init, |bbox, seg| Some(seg.transform(tr).bbox(bbox)))
            .expect("SubPath is never empty")
    }
}

/// Collection of the SubPath treated as a single unit. Represents the same concept
/// as an [SVG path](https://www.w3.org/TR/SVG11/paths.html)
#[derive(Clone, PartialEq, Default)]
pub struct Path {
    segments: Vec<Segment>,
    /// segments[subpath[i]..subpath[i+1]] represents i-th subpath
    subpaths: Vec<usize>,
    closed: Vec<bool>,
}

impl fmt::Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Path {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct FormatterWrite<'a, 'b> {
            fmt: &'a mut fmt::Formatter<'b>,
        }

        impl std::io::Write for FormatterWrite<'_, '_> {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.fmt
                    .write_str(
                        std::str::from_utf8(buf)
                            .expect("Path generated non utf8 svg representation"),
                    )
                    .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        self.write_svg_path(FormatterWrite { fmt })
            .map_err(|_| std::fmt::Error)
    }
}

impl Path {
    pub fn new(segments: Vec<Segment>, subpaths: Vec<usize>, closed: Vec<bool>) -> Self {
        Self {
            segments,
            subpaths,
            closed,
        }
    }

    /// Create empty path
    pub fn empty() -> Self {
        Self {
            segments: Vec::new(),
            subpaths: Vec::new(),
            closed: Vec::new(),
        }
    }

    /// Check if the path is empty
    pub fn is_empty(&self) -> bool {
        self.subpaths.is_empty()
    }

    /// Number of sub-paths in the path
    pub fn len(&self) -> usize {
        if self.subpaths.len() < 2 {
            0
        } else {
            self.subpaths.len() - 1
        }
    }

    /// Calculate winding number of a point
    pub fn winding_at(&self, point: impl Into<Point>) -> i32 {
        let point = point.into();
        // We are using horizontal line `y = point.y` to calculate winding number
        // - Find all segments that can potentially intersect this line.
        //   If all control points are on one side of the line then it is not going to
        //   intersect it as bezier curve is always bound by all its control points.
        // - Find intersection and based on tangent direction assign 1 or -1, throw away
        //   all intersections with `x > point.x`
        let y = point.y();
        let tr = Transform::new_translate(0.0, -y);
        let mut winding = 0;
        for subpath in self.subpaths() {
            let last = if subpath.closed {
                Some(Line::new(subpath.end(), subpath.start()).into())
            } else {
                None
            };
            for segment in subpath.segments().iter().chain(&last) {
                let points: &[Point] = match segment {
                    Segment::Line(Line(points)) => points,
                    Segment::Quad(Quad(points)) => points,
                    Segment::Cubic(Cubic(points)) => points,
                };
                let (above, below) = points.iter().fold((false, false), |(above, below), ctrl| {
                    if ctrl.y() > point.y() {
                        (true, below)
                    } else {
                        (above, true)
                    }
                });
                if !above || !below {
                    // will not intersect horizontal line
                    continue;
                }
                for root_t in segment.transform(tr).roots() {
                    let root = segment.at(root_t);
                    if root.x() > point.x() {
                        continue;
                    }
                    let deriv = segment.deriv().at(root_t);
                    if deriv.y() > 0.0 {
                        winding += 1;
                    } else if deriv.y() < 0.0 {
                        winding -= 1;
                    }
                }
            }
        }
        winding
    }

    /// Get sub-path
    pub fn get(&self, index: usize) -> Option<SubPath<'_>> {
        let start = *self.subpaths.get(index)?;
        let end = *self.subpaths.get(index + 1)?;
        let segments = self.segments.get(start..end)?;
        let closed = *self.closed.get(index)?;
        Some(SubPath { segments, closed })
    }

    /// List of sub-paths
    pub fn subpaths(&self) -> PathIter<'_> {
        PathIter {
            path: self,
            index: 0,
        }
    }

    pub fn push(&mut self, segments: &[Segment], closed: bool) {
        if segments.is_empty() {
            return;
        }
        if self.subpaths.is_empty() {
            self.subpaths.push(0);
        }
        self.segments.extend(segments.iter().copied());
        self.subpaths.push(self.segments.len());
        self.closed.push(closed);
    }

    /// Convenience method to create [`PathBuilder`]
    pub fn builder() -> PathBuilder {
        PathBuilder::new()
    }

    /// Convert path into a path builder so it can be extended
    pub fn into_builder(self) -> PathBuilder {
        PathBuilder::from_path(self)
    }

    /// Apply transformation to the path in place
    pub fn transform(&mut self, tr: Transform) {
        self.segments.iter_mut().for_each(|segment| {
            *segment = segment.transform(tr);
        });
    }

    /// Number of segments in the path
    pub fn segments_count(&self) -> usize {
        self.segments.len()
    }

    /// Stroke path
    ///
    /// Stroked path is the path constructed from original by offsetting by `distance/2` and
    /// joining it with the path offset by `-distance/2`.
    pub fn stroke(&self, style: StrokeStyle) -> Path {
        let mut result = Path::empty();
        let mut segments = Vec::new();
        for subpath in self.subpaths() {
            // forward
            for segment in subpath.segments() {
                stroke_segment(&mut segments, *segment, style, Segment::line_join);
            }
            let mut backward = subpath.segments.iter().rev().map(Segment::reverse);
            // close subpath
            if subpath.is_closed() {
                stroke_close(subpath, &mut segments, style, true);
                result.push(&segments, true);
                segments.clear();
            } else {
                // cap
                if let Some(segment) = backward.next() {
                    stroke_segment(&mut segments, segment, style, Segment::line_cap);
                }
            }
            // backward
            for segment in backward {
                stroke_segment(&mut segments, segment, style, Segment::line_join);
            }
            // close subpath
            if subpath.is_closed() {
                stroke_close(subpath, &mut segments, style, false);
                result.push(&segments, true);
                segments.clear();
            } else {
                // cap
                let last = segments.last().copied();
                let first = segments.first().copied();
                if let (Some(last), Some(first)) = (last, first) {
                    segments.extend(last.line_cap(first, style));
                }
                result.push(&segments, true);
                segments.clear();
            }
        }
        result
    }

    /// Convert path to an iterator over line segments
    pub fn flatten(
        &self,
        tr: Transform,
        flatness: Scalar,
        close: bool,
    ) -> impl Iterator<Item = Line> + '_ {
        PathFlattenIter::new(self, tr, flatness, close)
    }

    /// Bounding box of the path after provided transformation is applied.
    pub fn bbox(&self, tr: Transform) -> Option<BBox> {
        self.subpaths()
            .fold(None, |bbox, subpath| Some(subpath.bbox(bbox, tr)))
    }

    /// Calculate size of the image required to render the path
    ///
    /// Returns:
    ///   - Size of the image
    ///   - Transformation required
    ///   - Position of lowest x and y point of the image
    pub fn size(&self, tr: Transform) -> Option<(Size, Transform, Point)> {
        let bbox = self.bbox(tr)?;
        let Point([min_x, min_y]) = bbox.min();
        let Point([max_x, max_y]) = bbox.max();
        let min = Point::new(min_x.floor() - 1.0, min_y.floor() - 1.0);
        let max = Point::new(max_x.ceil() + 1.0, max_y.ceil() + 1.0);
        let size = Size {
            width: (max.x() - min.x()).round() as usize,
            height: (max.y() - min.y()).round() as usize,
        };
        let shift = Transform::new_translate(1.0 - min_x, 1.0 - min_y);
        Some((size, shift * tr, min))
    }

    /// Reverse order and direction of all segments
    pub fn reverse(&mut self) {
        if self.segments.is_empty() {
            return;
        }

        // reverse segments
        let mut left = 0;
        let mut right = self.segments.len() - 1;
        while left < right {
            let left_segment = self.segments[left].reverse();
            let right_segment = self.segments[right].reverse();
            self.segments[left] = right_segment;
            self.segments[right] = left_segment;
            left += 1;
            right -= 1;
        }
        if left == right {
            let left_segment = self.segments[left].reverse();
            self.segments[left] = left_segment;
        }

        // reverse sub-paths offsets
        for index in 0..(self.subpaths.len() - 1) {
            self.subpaths[index] = self.subpaths[index + 1] - self.subpaths[index];
        }
        self.subpaths.reverse();
        self.subpaths[0] = 0;
        let mut offset = 0;
        for index in 1..self.subpaths.len() {
            offset += self.subpaths[index];
            self.subpaths[index] = offset;
        }

        // reverse closed
        self.closed.reverse();
    }

    /// Fill path with the provided paint
    pub fn fill<R, P, I>(
        &self,
        rasterizer: R,
        tr: Transform,
        fill_rule: FillRule,
        paint: P,
        mut img: I,
    ) -> I
    where
        R: Rasterizer,
        P: Paint,
        I: ImageMut<Pixel = LinColor>,
    {
        rasterizer.fill(self, tr, fill_rule, &paint, &mut img);
        img
    }

    /// Rasterize mast for the path in into a provided image.
    ///
    /// Everything that is outside of the image will be cropped. Image is assumed
    /// to contain zeros.
    pub fn mask<R, I>(&self, rasterizer: R, tr: Transform, fill_rule: FillRule, mut img: I) -> I
    where
        R: Rasterizer,
        I: ImageMut<Pixel = Scalar>,
    {
        rasterizer.mask(self, tr, &mut img, fill_rule);
        img
    }

    /// Save path in SVG path format.
    pub fn write_svg_path(&self, mut out: impl Write) -> std::io::Result<()> {
        for subpath in self.subpaths() {
            write!(out, "M{:?} ", subpath.start())?;
            let mut segment_type: Option<u8> = None;
            for segment in subpath.segments().iter() {
                match segment {
                    Segment::Line(line) => {
                        if segment_type.replace(b'L') != Some(b'L') {
                            out.write_all(b"L")?;
                        }
                        write!(out, "{:?} ", line.end())?;
                    }
                    Segment::Quad(quad) => {
                        let [_, p1, p2] = quad.points();
                        if segment_type.replace(b'Q') != Some(b'Q') {
                            out.write_all(b"Q")?;
                        }
                        write!(out, "{:?} {:?} ", p1, p2)?;
                    }
                    Segment::Cubic(cubic) => {
                        let [_, p1, p2, p3] = cubic.points();
                        if segment_type.replace(b'C') != Some(b'C') {
                            out.write_all(b"C")?;
                        }
                        write!(out, "{:?} {:?} {:?} ", p1, p2, p3)?;
                    }
                }
            }
            if subpath.is_closed() {
                out.write_all(b"Z")?;
            }
        }
        Ok(())
    }

    /// Load path from SVG path representation
    pub fn read_svg_path(input: impl Read) -> std::io::Result<Self> {
        let mut builder = PathBuilder::new();
        for cmd in SvgPathParser::new(input) {
            cmd?.apply(&mut builder)
        }
        Ok(builder.build())
    }

    /// Returns struct that prints command per line on debug formatting.
    pub fn verbose_debug(&self) -> PathVerboseDebug<'_> {
        PathVerboseDebug { path: self }
    }
}

pub struct PathVerboseDebug<'a> {
    path: &'a Path,
}

impl fmt::Debug for PathVerboseDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.subpaths.is_empty() {
            write!(f, "Empty")?;
        } else {
            for subpath in self.path.subpaths.iter() {
                subpath.fmt(f)?
            }
        }
        Ok(())
    }
}

pub struct PathIter<'a> {
    path: &'a Path,
    index: usize,
}

impl<'a> Iterator for PathIter<'a> {
    type Item = SubPath<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.path.get(self.index)?;
        self.index += 1;
        Some(result)
    }
}

impl<'a> IntoIterator for &'a Path {
    type Item = SubPath<'a>;
    type IntoIter = PathIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.subpaths()
    }
}

impl<'a> Extend<SubPath<'a>> for Path {
    fn extend<T: IntoIterator<Item = SubPath<'a>>>(&mut self, iter: T) {
        for subpath in iter {
            self.push(&subpath.segments, subpath.closed);
        }
    }
}

/// Extend segments with the offset segment and join between those segments.
fn stroke_segment<F, S>(segments: &mut Vec<Segment>, segment: Segment, style: StrokeStyle, join: F)
where
    F: Fn(Segment, Segment, StrokeStyle) -> S,
    S: IntoIterator<Item = Segment>,
{
    let offset = segments.len();
    segment.offset(style.width / 2.0, segments);
    if offset != 0 {
        let src = segments.get(offset - 1).copied();
        let dst = segments.get(offset).copied();
        if let (Some(src), Some(dst)) = (src, dst) {
            segments.splice(offset..offset, join(src, dst, style));
        }
    }
}

fn stroke_close<'a>(
    subpath: SubPath<'a>,
    segments: &mut Vec<Segment>,
    style: StrokeStyle,
    forward: bool,
) {
    let (first, last) = match (segments.first(), segments.last()) {
        (Some(first), Some(last)) => (*first, *last),
        _ => return,
    };
    let close = if forward {
        Line::new(subpath.end(), subpath.start())
    } else {
        Line::new(subpath.start(), subpath.end())
    };
    match line_offset(close, style.width / 2.0) {
        Some(close) if close.length() * 100.0 > style.width => {
            let close = Segment::from(close);
            segments.extend(last.line_join(close, style));
            segments.push(close);
            segments.extend(close.line_join(first, style));
        }
        _ => segments.extend(last.line_join(first, style)),
    }
}

pub struct PathFlattenIter<'a> {
    path: &'a Path,
    transform: Transform,
    flatness: Scalar,
    close: bool,
    subpath_index: usize,
    segment_index: usize,
    stack: Vec<Segment>,
}

impl<'a> PathFlattenIter<'a> {
    fn new(path: &'a Path, transform: Transform, flatness: Scalar, close: bool) -> Self {
        Self {
            path,
            transform,
            flatness: 16.0 * flatness * flatness,
            close,
            subpath_index: 0,
            segment_index: 0,
            stack: Default::default(),
        }
    }
}

impl Iterator for PathFlattenIter<'_> {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stack.pop() {
                Some(segment) => {
                    if segment.has_nans() {
                        panic!("cannot flatten segment with NaN");
                    }
                    if segment.flatness() < self.flatness {
                        return Some(Line::new(segment.start(), segment.end()));
                    }
                    let (s0, s1) = segment.split();
                    self.stack.push(s1);
                    self.stack.push(s0);
                }
                None => {
                    let subpath = self.path.get(self.subpath_index)?;
                    match subpath.segments().get(self.segment_index) {
                        None => {
                            self.subpath_index += 1;
                            self.segment_index = 0;
                            if subpath.closed || self.close {
                                let line = Line::new(subpath.end(), subpath.start())
                                    .transform(self.transform);
                                return Some(line);
                            }
                        }
                        Some(segment) => {
                            self.segment_index += 1;
                            self.stack.push(segment.transform(self.transform));
                        }
                    }
                }
            }
        }
    }
}

/// Path builder similar to Canvas/Cairo interface.
#[derive(Clone)]
pub struct PathBuilder {
    position: Point,
    segments: Vec<Segment>,
    subpaths: Vec<usize>,
    closed: Vec<bool>,
}

impl Default for PathBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PathBuilder {
    pub fn new() -> Self {
        Self {
            position: Point::new(0.0, 0.0),
            segments: Vec::new(),
            subpaths: Vec::new(),
            closed: Vec::new(),
        }
    }

    pub fn from_path(path: Path) -> Self {
        let mut builder = Self::new();
        builder.segments = path.segments;
        builder.subpaths = path.subpaths;
        builder.closed = path.closed;
        builder
    }

    /// Build path
    pub fn build(&mut self) -> Path {
        self.subpath_finish(false);
        let PathBuilder {
            segments,
            subpaths,
            closed,
            ..
        } = std::mem::take(self);
        Path::new(segments, subpaths, closed)
    }

    pub fn push(&mut self, segments: &[Segment], closed: bool) {
        self.segments.extend(segments.iter().copied());
        self.subpath_finish(closed);
    }

    /// Finish current subpath
    fn subpath_finish(&mut self, close: bool) {
        if self.segments.is_empty() || self.subpaths.last().copied() == Some(self.segments.len()) {
            return;
        }
        if self.subpaths.is_empty() {
            self.subpaths.push(0);
        }
        if close {
            // if we close subpath, current position is set to start of the current subpath
            if let Some(subpath_start) = self
                .subpaths
                .last()
                .and_then(|first_index| Some(self.segments.get(*first_index)?.start()))
            {
                self.position = subpath_start;
            }
        }
        self.subpaths.push(self.segments.len());
        self.closed.push(close);
    }

    /// Extend path from string, which is specified in the same format as SVGs path element.
    pub fn append_svg_path(
        &mut self,
        string: impl AsRef<[u8]>,
    ) -> Result<&mut Self, SvgParserError> {
        for cmd in SvgPathParser::new(Cursor::new(string)) {
            cmd?.apply(self);
        }
        Ok(self)
    }

    /// Move current position, ending current subpath
    pub fn move_to(&mut self, p: impl Into<Point>) -> &mut Self {
        self.subpath_finish(false);
        self.position = p.into();
        self
    }

    /// Close current subpath
    pub fn close(&mut self) -> &mut Self {
        self.subpath_finish(true);
        self
    }

    /// Add line from the current position to the specified point
    pub fn line_to(&mut self, p: impl Into<Point>) -> &mut Self {
        let p = p.into();
        if !self.position.is_close_to(p) {
            let line = Line::new(self.position, p);
            self.position = line.end();
            self.segments.push(line.into());
        }
        self
    }

    /// Add quadratic bezier curve
    pub fn quad_to(&mut self, p1: impl Into<Point>, p2: impl Into<Point>) -> &mut Self {
        let quad = Quad::new(self.position, p1, p2);
        self.position = quad.end();
        self.segments.push(quad.into());
        self
    }

    /// Add smooth quadratic bezier curve
    pub fn quad_smooth_to(&mut self, p2: impl Into<Point>) -> &mut Self {
        let p1 = match self.segments.last() {
            Some(Segment::Quad(quad)) => quad.smooth(),
            _ => self.position,
        };
        self.quad_to(p1, p2)
    }

    /// Add cubic bezier curve
    pub fn cubic_to(
        &mut self,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
        p3: impl Into<Point>,
    ) -> &mut Self {
        let cubic = Cubic::new(self.position, p1, p2, p3);
        self.position = cubic.end();
        self.segments.push(cubic.into());
        self
    }

    /// Add smooth cubic bezier curve
    pub fn cubic_smooth_to(&mut self, p2: impl Into<Point>, p3: impl Into<Point>) -> &mut Self {
        let p1 = match self.segments.last() {
            Some(Segment::Cubic(cubic)) => cubic.smooth(),
            _ => self.position,
        };
        self.cubic_to(p1, p2, p3)
    }

    /// Add elliptic arc segment
    pub fn arc_to(
        &mut self,
        radii: impl Into<Point>,
        x_axis_rot: Scalar,
        large: bool,
        sweep: bool,
        p: impl Into<Point>,
    ) -> &mut Self {
        let radii: Point = radii.into();
        let p = p.into();
        let arc = EllipArc::new_param(
            self.position,
            p,
            radii.x(),
            radii.y(),
            x_axis_rot,
            large,
            sweep,
        );
        match arc {
            None => self.line_to(p),
            Some(arc) => {
                self.segments.extend(arc.to_cubics().map(Segment::from));
                self.position = p;
                self
            }
        }
    }

    /// Add circle with the center at current position and provided radius.
    ///
    /// Current position is not changed after invocation.
    pub fn circle(&mut self, radius: Scalar) -> &mut Self {
        // https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
        // (4/3)*tan(pi/8) = 4*(sqrt(2)-1)/3 = 0.5522847498307935
        let offset = 0.5522847498307935 * radius;
        let x_offset = Point::new(offset, 0.0);
        let y_offset = Point::new(0.0, offset);
        let center = self.position();
        let p0 = center - Point::new(radius, 0.0);
        let p1 = center - Point::new(0.0, radius);
        let p2 = center + Point::new(radius, 0.0);
        let p3 = center + Point::new(0.0, radius);

        self.move_to(p0)
            .cubic_to(p0 - y_offset, p1 - x_offset, p1)
            .cubic_to(p1 + x_offset, p2 - y_offset, p2)
            .cubic_to(p2 + y_offset, p3 + x_offset, p3)
            .cubic_to(p3 - x_offset, p0 + y_offset, p0)
            .close()
            .move_to(center)
    }

    /// Add box with rounded corners, with current position being low-x and low-y coordinate
    pub fn rbox(&mut self, size: impl Into<Point>, radii: impl Into<Point>) -> &mut Self {
        let init = self.position;
        let bbox = BBox::new(self.position, self.position + size.into());
        let Point([lx, ly]) = bbox.min();
        let Point([hx, hy]) = bbox.max();

        let Point([rx, ry]) = radii.into();
        let rx = clamp(rx.abs(), 0.0, hx - lx);
        let ry = clamp(ry.abs(), 0.0, hy - ly);
        let radii = Point::new(rx, ry);
        let rounded = rx > EPSILON && ry > EPSILON;

        self.move_to((lx + rx, ly)).line_to((hx - rx, ly));
        if rounded {
            self.arc_to(radii, 0.0, false, true, (hx, ly + ry));
        }
        self.line_to((hx, hy - ry));
        if rounded {
            self.arc_to(radii, 0.0, false, true, (hx - rx, hy));
        }
        self.line_to((lx + rx, hy));
        if rounded {
            self.arc_to(radii, 0.0, false, true, (lx, hy - ry));
        }
        self.line_to((lx, ly + ry));
        if rounded {
            self.arc_to(radii, 0.0, false, true, (lx + rx, ly));
        }
        self.close().move_to(init)
    }

    /// Create checker board path inside bounding, useful to draw transparent area
    pub fn checkerboard(&mut self, bbox: BBox, cell_size: Scalar) -> &mut Self {
        let mut x = bbox.x();
        let mut y = bbox.y();
        while y < bbox.max().y() {
            while x < bbox.max().x() {
                let offset = Point::new(x, y);
                self.move_to(offset)
                    .line_to(offset + Point::new(cell_size, 0.0))
                    .line_to(offset + Point::new(cell_size, 2.0 * cell_size))
                    .line_to(offset + Point::new(2.0 * cell_size, 2.0 * cell_size))
                    .line_to(offset + Point::new(2.0 * cell_size, cell_size))
                    .line_to(offset + Point::new(0.0, cell_size))
                    .close();
                x += 2.0 * cell_size;
            }
            x = bbox.x();
            y += 2.0 * cell_size;
        }
        self.move_to(bbox.min())
    }

    /// Current position of the builder
    pub fn position(&self) -> Point {
        self.position
    }
}

impl FromStr for Path {
    type Err = SvgParserError;

    fn from_str(text: &str) -> Result<Path, Self::Err> {
        let mut builder = PathBuilder::new();
        for cmd in SvgPathParser::new(Cursor::new(text)) {
            cmd?.apply(&mut builder);
        }
        Ok(builder.build())
    }
}

#[cfg(feature = "serde")]
impl Serialize for Path {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Path {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        std::borrow::Cow::<'de, str>::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq, PI};

    fn assert_path_eq(p0: &Path, p1: &Path) {
        assert_eq!(format!("{:?}", p0), format!("{:?}", p1));
    }

    #[test]
    fn test_bbox() {
        let path: Path = SQUIRREL.parse().unwrap();
        let bbox = path.bbox(Transform::identity()).unwrap();
        assert_approx_eq!(bbox.x(), 0.25);
        assert_approx_eq!(bbox.y(), 1.0);
        assert_approx_eq!(bbox.width(), 15.75);
        assert_approx_eq!(bbox.height(), 14.0);
    }

    const SQUIRREL: &str = r#"
    M12 1C9.79 1 8 2.31 8 3.92c0 1.94.5 3.03 0 6.08 0-4.5-2.77-6.34-4-6.34.05-.5-.48
    -.66-.48-.66s-.22.11-.3.34c-.27-.31-.56-.27-.56-.27l-.13.58S.7 4.29 .68 6.87c.2.33
    1.53.6 2.47.43.89.05.67.79.47.99C2.78 9.13 2 8 1 8S0 9 1 9s1 1 3 1c-3.09 1.2 0 4 0 4
    H3c-1 0-1 1-1 1h6c3 0 5-1 5-3.47 0-.85-.43-1.79 -1-2.53-1.11-1.46.23-2.68 1-2
    .77.68 3 1 3-2 0-2.21-1.79-4-4-4zM2.5 6 c-.28 0-.5-.22-.5-.5s.22-.5.5-.5.5.22.5.5
    -.22.5-.5.5z
    "#;

    #[test]
    fn test_path_parse() -> Result<(), SvgParserError> {
        // complicated path
        let path: Path = SQUIRREL.parse()?;
        let reference = Path::builder()
            .move_to((12.0, 1.0))
            .cubic_to((9.79, 1.0), (8.0, 2.31), (8.0, 3.92))
            .cubic_to((8.0, 5.86), (8.5, 6.95), (8.0, 10.0))
            .cubic_to((8.0, 5.5), (5.23, 3.66), (4.0, 3.66))
            .cubic_to((4.05, 3.16), (3.52, 3.0), (3.52, 3.0))
            .cubic_to((3.52, 3.0), (3.3, 3.11), (3.22, 3.34))
            .cubic_to((2.95, 3.03), (2.66, 3.07), (2.66, 3.07))
            .line_to((2.53, 3.65))
            .cubic_to((2.53, 3.65), (0.7, 4.29), (0.68, 6.87))
            .cubic_to((0.88, 7.2), (2.21, 7.47), (3.15, 7.3))
            .cubic_to((4.04, 7.35), (3.82, 8.09), (3.62, 8.29))
            .cubic_to((2.78, 9.13), (2.0, 8.0), (1.0, 8.0))
            .cubic_to((0.0, 8.0), (0.0, 9.0), (1.0, 9.0))
            .cubic_to((2.0, 9.0), (2.0, 10.0), (4.0, 10.0))
            .cubic_to((0.91, 11.2), (4.0, 14.0), (4.0, 14.0))
            .line_to((3.0, 14.0))
            .cubic_to((2.0, 14.0), (2.0, 15.0), (2.0, 15.0))
            .line_to((8.0, 15.0))
            .cubic_to((11.0, 15.0), (13.0, 14.0), (13.0, 11.53))
            .cubic_to((13.0, 10.68), (12.57, 9.74), (12.0, 9.0))
            .cubic_to((10.89, 7.54), (12.23, 6.32), (13.0, 7.0))
            .cubic_to((13.77, 7.68), (16.0, 8.0), (16.0, 5.0))
            .cubic_to((16.0, 2.79), (14.21, 1.0), (12.0, 1.0))
            .close()
            .move_to((2.5, 6.0))
            .cubic_to((2.22, 6.0), (2.0, 5.78), (2.0, 5.5))
            .cubic_to((2.0, 5.22), (2.22, 5.0), (2.5, 5.0))
            .cubic_to((2.78, 5.0), (3.0, 5.22), (3.0, 5.5))
            .cubic_to((3.0, 5.78), (2.78, 6.0), (2.5, 6.0))
            .close()
            .build();
        assert_path_eq(&path, &reference);

        let path: Path = " M0,0L1-1L1,0ZL0,1 L1,1Z ".parse()?;
        let reference = Path::builder()
            .move_to((0.0, 0.0))
            .line_to((1.0, -1.0))
            .line_to((1.0, 0.0))
            .close()
            .move_to((0.0, 0.0))
            .line_to((0.0, 1.0))
            .line_to((1.0, 1.0))
            .close()
            .build();
        assert_path_eq(&path, &reference);

        let reference = Path::builder()
            .move_to((0.5, -3.0))
            .line_to((-11.0, -0.11))
            .build();
        // not separated scalars, implicit line segment
        let p1: Path = "M.5-3-11-.11".parse()?;
        // other spaces, implicit relative line segment
        let p2: Path = " m.5,-3 -11.5\n2.89 ".parse()?;
        assert_path_eq(&reference, &p1);
        assert_path_eq(&reference, &p2);

        Ok(())
    }

    #[test]
    fn test_save_load() -> std::io::Result<()> {
        let path: Path = SQUIRREL.parse()?;
        let mut path_save = Vec::new();
        path.write_svg_path(&mut path_save)?;
        let path_load = Path::read_svg_path(std::io::Cursor::new(path_save))?;
        assert_path_eq(&path, &path_load);
        Ok(())
    }

    #[test]
    fn test_flatten() -> Result<(), SvgParserError> {
        let path: Path = SQUIRREL.parse()?;
        let tr = Transform::new_rotate(PI / 3.0).pre_translate(-10.0, -20.0);
        let lines: Vec<_> = path.flatten(tr, DEFAULT_FLATNESS, true).collect();
        let mut reference = Vec::new();
        for subpath in path.subpaths() {
            let subpath_lines: Vec<_> = subpath.flatten(tr, DEFAULT_FLATNESS, false).collect();
            // line are connected
            for ls in subpath_lines.windows(2) {
                assert!(ls[0].end().is_close_to(ls[1].start()));
            }
            reference.extend(subpath_lines);
        }
        assert_eq!(reference.len(), lines.len());
        for (l0, l1) in lines.iter().zip(reference.iter()) {
            assert!(l0.start().is_close_to(l1.start()));
            assert!(l0.end().is_close_to(l1.end()));
        }
        Ok(())
    }

    #[test]
    fn test_winding() -> Result<(), SvgParserError> {
        let path: Path = SQUIRREL.parse()?;
        assert_eq!(path.winding_at((2.4, 5.4)), 0);
        assert_eq!(path.winding_at((3.5, 3.155)), 1);
        assert_eq!(path.winding_at((3.92, 3.155)), 0);
        assert_eq!(path.winding_at((12.46, 6.87)), 0);
        assert_eq!(path.winding_at((14.24, 7.455)), 1);
        Ok(())
    }

    #[test]
    fn test_stroke() -> Result<(), SvgParserError> {
        // open path
        let path: Path = "M2,2L8,2C11,2 11,8 8,8L5,4".parse()?;

        let path_stroke = path.stroke(StrokeStyle {
            width: 1.0,
            ..Default::default()
        });
        let path_reference: Path = r#"
        M2,1.5 L8,1.5 C9.80902,1.5 10.75,3.38197 10.75,5 10.75,6.61803 9.80902,8.5 8,8.5 L7.75,8.5 7.6,8.3 4.6,4.3
        5.4,3.7 8.4,7.7 8,7.5 C9.19098,7.5 9.75,6.38197 9.75,5 9.75,3.61803 9.19098,2.5 8,2.5 L2,2.5 2,1.5 Z
        "#.parse()?;
        assert_path_eq(&path_reference, &path_stroke);

        let path_stroke = path.stroke(StrokeStyle {
            width: 1.0,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            ..Default::default()
        });
        let path_reference: Path = r#"
        M2,1.5 L8,1.5 C9.80902,1.5 10.75,3.38197 10.75,5 10.75,6.61803 9.80902,8.5 8,8.5 7.84274,8.5 7.69436,8.42581
        7.6,8.3 L4.6,4.3 C4.43542,4.08057 4.48057,3.76458 4.7,3.6 4.91943,3.43542 5.23542,3.48057 5.4,3.7 L8.4,7.7 8,7.5
        C9.19098,7.5 9.75,6.38197 9.75,5 9.75,3.61803 9.19098,2.5 8,2.5 L2,2.5 C1.72571,2.5 1.5,2.27429 1.5,2
        1.5,1.72571 1.72571,1.5 2,1.5 Z
        "#.parse()?;
        assert_path_eq(&path_reference, &path_stroke);

        // closed path
        let path: Path = "M2,2L8,2C11,2 11,8 8,8L5,4Z".parse()?;

        let path_stroke = path.stroke(StrokeStyle {
            width: 1.0,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            ..Default::default()
        });
        let path_reference: Path = r#"
        M2,1.5 L8,1.5 C9.80902,1.5 10.75,3.38197 10.75,5 10.75,6.61803 9.80902,8.5 8,8.5 7.84274,8.5 7.69436,8.42581
        7.6,8.3 L4.6,4.3 4.72265,4.41603 1.72265,2.41603 C1.53984,2.29415 1.45778,2.06539 1.52145,1.85511 1.58512,1.64482
        1.78029,1.5 2,1.5 ZM5.4,3.7 L8.4,7.7 8,7.5 C9.19098,7.5 9.75,6.38197 9.75,5 9.75,3.61803 9.19098,2.5 8,2.5
        L2,2.5 2.27735,1.58397 5.27735,3.58397 C5.32451,3.61542 5.36599,3.65465 5.4,3.7 Z
        "#.parse()?;
        assert_path_eq(&path_reference, &path_stroke);

        Ok(())
    }

    #[test]
    fn test_reverse() -> Result<(), SvgParserError> {
        let mut path: Path =
            "M2,2L8,2C11,2 11,8 8,8L5,4ZM7,3L8,6C9,6 10,5 9,4ZM2,4Q2,8 3,8L5,8C6,8 2,3 2,4Z"
                .parse()?;
        path.reverse();
        let path_reference =
            "M2,4C2,3 6,8 5,8L3,8Q2,8 2,4ZM9,4C10,5 9,6 8,6L7,3ZM5,4L8,8C11,8 11,2 8,2L2,2Z"
                .parse()?;
        assert_path_eq(&path_reference, &path);
        Ok(())
    }
}
