use crate::{
    clamp, curve::line_offset, rasterize::Rasterizer, BBox, Brush, Cubic, Curve, EllipArc,
    ImageMut, LinColor, Line, Point, Quad, SVGPathParser, SVGPathParserError, Scalar, Segment,
    Size, Transform, EPSILON,
};
use std::{
    fmt,
    io::{Cursor, Read, Write},
    str::FromStr,
    usize,
};

/// Default flatness used during rasterizetion.
/// Value of 0.05px gives good accuracy tradeoff.
pub const DEFAULT_FLATNESS: Scalar = 0.05;

/// The algorithm to use to determine the inside part of a shape, when filling it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FillRule {
    /// Fill area with non-zero winding number
    NonZero,
    /// Fill area with odd winding number
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

impl Default for FillRule {
    fn default() -> Self {
        FillRule::NonZero
    }
}

/// `LineJoin` defines the shape to be used at the corners of paths when they are stroked.
/// See [SVG specification](https://www.w3.org/TR/SVG2/painting.html#LineJoin) for more details.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum LineJoin {
    /// Continue path segments with lines untill they intersect. But only
    /// if `miter_length = stroke-width / sin(0.5 * eta)` is less than the miter argument.
    Miter(Scalar),
    /// Connect path segments with straigh line.
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

/// `LineCap` specifies the shape to be used at the end of open subpaths when they are stroked.
/// See [SVG specification](https://www.w3.org/TR/SVG2/painting.html#LineCaps) for more details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct StrokeStyle {
    /// Width of the stroke
    pub width: Scalar,
    /// How to join offset segments
    pub line_join: LineJoin,
    /// How to join segments at the ends of the path
    pub line_cap: LineCap,
}

/// Non-empty collections of segments where end of each segments conisides with the start of the next one.
#[derive(Clone, PartialEq)]
pub struct SubPath {
    /// List of segments representing SubPath
    segments: Vec<Segment>,
    /// Whether SubPath contains an implicit line segment connecting start and the end of it.
    closed: bool,
}

impl fmt::Debug for SubPath {
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

impl SubPath {
    pub fn new(segments: Vec<Segment>, closed: bool) -> Option<Self> {
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

    /// Apply transformation to the sub-path in place
    pub fn transform(&mut self, tr: Transform) {
        for segment in self.segments.iter_mut() {
            *segment = segment.transform(tr);
        }
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

    /// Create new sub-path with reversed direction
    pub fn reverse(&self) -> Self {
        Self {
            segments: self.segments.iter().rev().map(|s| s.reverse()).collect(),
            closed: self.closed,
        }
    }
}

/// Collection of the SubPath treated as a signle unit
#[derive(Clone, PartialEq, Default)]
pub struct Path {
    subpaths: Vec<SubPath>,
}

impl fmt::Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.subpaths.is_empty() {
            write!(f, "Empty")?;
        } else {
            for subpath in self.subpaths.iter() {
                subpath.fmt(f)?
            }
        }
        Ok(())
    }
}

impl Path {
    /// Create path from the list of subpaths
    pub fn new(subpaths: Vec<SubPath>) -> Self {
        Self { subpaths }
    }

    /// Create empty path
    pub fn empty() -> Self {
        Self {
            subpaths: Default::default(),
        }
    }

    pub fn subpaths(&self) -> &[SubPath] {
        &self.subpaths
    }

    /// Convenience method to create `PathBuilder`
    pub fn builder() -> PathBuilder {
        PathBuilder::new()
    }

    pub fn into_builder(self) -> PathBuilder {
        PathBuilder::from_path(self)
    }

    /// Apply transformation to the path in place
    pub fn transform(&mut self, tr: Transform) {
        for subpath in self.subpaths.iter_mut() {
            subpath.transform(tr);
        }
    }

    pub fn segments_count(&self) -> usize {
        self.subpaths
            .iter()
            .fold(0usize, |acc, subpath| acc + subpath.segments().len())
    }

    /// Stroke path
    ///
    /// Stroked path is the path constructed from original by offsetting by `distance/2` and
    /// joinging it with the path offsetted by `-distance/2`.
    pub fn stroke(&self, style: StrokeStyle) -> Path {
        let mut subpaths = Vec::new();
        for subpath in self.subpaths.iter() {
            let mut segments = Vec::new();
            // forward
            for segment in subpath.segments().iter() {
                stroke_segment(&mut segments, *segment, style, Segment::line_join);
            }
            let mut backward = subpath.segments.iter().rev().map(Segment::reverse);
            // close subpath
            if subpath.is_closed() {
                let segments = stroke_close(subpath, &mut segments, style, true);
                subpaths.extend(SubPath::new(segments, true));
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
                let segments = stroke_close(subpath, &mut segments, style, false);
                subpaths.extend(SubPath::new(segments, true));
            } else {
                // cap
                let last = segments.last().copied();
                let first = segments.first().copied();
                if let (Some(last), Some(first)) = (last, first) {
                    segments.extend(last.line_cap(first, style));
                }
                subpaths.extend(SubPath::new(segments, /* closed = */ true));
            }
        }
        Path::new(subpaths)
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
        self.subpaths
            .iter()
            .fold(None, |bbox, subpath| Some(subpath.bbox(bbox, tr)))
    }

    /// Calclulate size of the image required to render the path
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
        let shift = Transform::default().translate(1.0 - min_x, 1.0 - min_y);
        Some((size, shift * tr, min))
    }

    /// Rreverse order and direction of all segments
    pub fn reverse(&self) -> Self {
        Self {
            subpaths: self.subpaths.iter().map(|s| s.reverse()).collect(),
        }
    }

    /// Fill path with the provided color
    pub fn fill<R, B, I>(
        &self,
        rasterizer: R,
        tr: Transform,
        fill_rule: FillRule,
        brush: B,
        mut img: I,
    ) -> I
    where
        R: Rasterizer,
        B: Brush,
        I: ImageMut<Pixel = LinColor>,
    {
        rasterizer.fill(self, tr, fill_rule, &brush, &mut img);
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

    /// Convert path to SVG path representation
    pub fn to_svg_path(&self) -> String {
        let mut output = Vec::new();
        self.write_svg_path(&mut output)
            .expect("failed in memory write");
        String::from_utf8(output).expect("path save internal error")
    }

    /// Save path in SVG path format.
    pub fn write_svg_path(&self, mut out: impl Write) -> std::io::Result<()> {
        for subpath in self.subpaths.iter() {
            write!(&mut out, "M{:?} ", subpath.start())?;
            let mut segment_type: Option<u8> = None;
            for segment in subpath.segments().iter() {
                match segment {
                    Segment::Line(line) => {
                        if segment_type.replace(b'L') != Some(b'L') {
                            out.write_all(b"L")?;
                        }
                        write!(&mut out, "{:?} ", line.end())?;
                    }
                    Segment::Quad(quad) => {
                        let [_, p1, p2] = quad.points();
                        if segment_type.replace(b'Q') != Some(b'Q') {
                            out.write_all(b"Q")?;
                        }
                        write!(&mut out, "{:?} {:?} ", p1, p2)?;
                    }
                    Segment::Cubic(cubic) => {
                        let [_, p1, p2, p3] = cubic.points();
                        if segment_type.replace(b'C') != Some(b'C') {
                            out.write_all(b"C")?;
                        }
                        write!(&mut out, "{:?} {:?} {:?} ", p1, p2, p3)?;
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
        for cmd in SVGPathParser::new(input) {
            cmd?.apply(&mut builder)
        }
        Ok(builder.build())
    }
}

impl IntoIterator for Path {
    type Item = SubPath;
    type IntoIter = <Vec<SubPath> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.subpaths.into_iter()
    }
}

impl<'a> IntoIterator for &'a Path {
    type Item = &'a SubPath;
    type IntoIter = <&'a Vec<SubPath> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.subpaths.iter()
    }
}

impl Extend<SubPath> for Path {
    fn extend<T: IntoIterator<Item = SubPath>>(&mut self, iter: T) {
        self.subpaths.extend(iter)
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

fn stroke_close(
    subpath: &SubPath,
    segments: &mut Vec<Segment>,
    style: StrokeStyle,
    forward: bool,
) -> Vec<Segment> {
    let (first, last) = match (segments.first(), segments.last()) {
        (Some(first), Some(last)) => (*first, *last),
        _ => return Vec::new(),
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
    std::mem::replace(segments, Vec::new())
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

impl<'a> Iterator for PathFlattenIter<'a> {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stack.pop() {
                Some(segment) => {
                    if segment.flatness() < self.flatness {
                        return Some(Line::new(segment.start(), segment.end()));
                    }
                    let (s0, s1) = segment.split();
                    self.stack.push(s1);
                    self.stack.push(s0);
                }
                None => {
                    let subpath = self.path.subpaths.get(self.subpath_index)?;
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
    subpath: Vec<Segment>,
    subpaths: Vec<SubPath>,
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
            subpath: Default::default(),
            subpaths: Default::default(),
        }
    }

    pub fn from_path(path: Path) -> Self {
        let mut builder = Self::new();
        builder.subpaths = path.subpaths;
        builder
    }

    /// Build path
    pub fn build(&mut self) -> Path {
        let PathBuilder {
            subpath,
            mut subpaths,
            ..
        } = std::mem::take(self);
        subpaths.extend(SubPath::new(subpath, false));
        Path::new(subpaths)
    }

    /// Extend path from string, which is specified in the same format as SVGs path element.
    pub fn append_svg_path(
        &mut self,
        string: impl AsRef<[u8]>,
    ) -> Result<&mut Self, SVGPathParserError> {
        for cmd in SVGPathParser::new(Cursor::new(string)) {
            cmd?.apply(self);
        }
        Ok(self)
    }

    /// Move current position, ending current subpath
    pub fn move_to(&mut self, p: impl Into<Point>) -> &mut Self {
        let subpath = std::mem::replace(&mut self.subpath, Vec::new());
        self.subpaths.extend(SubPath::new(subpath, false));
        self.position = p.into();
        self
    }

    /// Close current subpath
    pub fn close(&mut self) -> &mut Self {
        let subpath = std::mem::replace(&mut self.subpath, Vec::new());
        if let Some(seg) = subpath.first() {
            self.position = seg.start();
        }
        self.subpaths.extend(SubPath::new(subpath, true));
        self
    }

    /// Add line from the current position to the specified point
    pub fn line_to(&mut self, p: impl Into<Point>) -> &mut Self {
        let p = p.into();
        if !self.position.is_close_to(p) {
            let line = Line::new(self.position, p);
            self.position = line.end();
            self.subpath.push(line.into());
        }
        self
    }

    /// Add quadratic bezier curve
    pub fn quad_to(&mut self, p1: impl Into<Point>, p2: impl Into<Point>) -> &mut Self {
        let quad = Quad::new(self.position, p1, p2);
        self.position = quad.end();
        self.subpath.push(quad.into());
        self
    }

    /// Add smooth quadratic bezier curve
    pub fn quad_smooth_to(&mut self, p2: impl Into<Point>) -> &mut Self {
        let p1 = match self.subpath.last() {
            Some(Segment::Quad(quad)) => quad.smooth(),
            _ => self.position,
        };
        self.quad_to(p1, p2)
    }

    /// Add cubic beizer curve
    pub fn cubic_to(
        &mut self,
        p1: impl Into<Point>,
        p2: impl Into<Point>,
        p3: impl Into<Point>,
    ) -> &mut Self {
        let cubic = Cubic::new(self.position, p1, p2, p3);
        self.position = cubic.end();
        self.subpath.push(cubic.into());
        self
    }

    /// Add smooth cubic bezier curve
    pub fn cubic_smooth_to(&mut self, p2: impl Into<Point>, p3: impl Into<Point>) -> &mut Self {
        let p1 = match self.subpath.last() {
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
                self.subpath.extend(arc.to_cubics().map(Segment::from));
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

    /// Current possition of the builder
    pub fn position(&self) -> Point {
        self.position
    }
}

impl FromStr for Path {
    type Err = SVGPathParserError;

    fn from_str(text: &str) -> Result<Path, Self::Err> {
        let mut builder = PathBuilder::new();
        for cmd in SVGPathParser::new(Cursor::new(text)) {
            cmd?.apply(&mut builder);
        }
        Ok(builder.build())
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
        let bbox = path.bbox(Transform::default()).unwrap();
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
    fn test_path_parse() -> Result<(), SVGPathParserError> {
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
        // unseparated scalars, implicit line segment
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
    fn test_flatten() -> Result<(), SVGPathParserError> {
        let path: Path = SQUIRREL.parse()?;
        let tr = Transform::default()
            .rotate(PI / 3.0)
            .translate(-10.0, -20.0);
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
}
