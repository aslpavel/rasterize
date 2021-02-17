use crate::{
    curve::line_offset,
    rasterize::{signed_difference_line, signed_difference_to_mask},
    Align, BBox, Cubic, Curve, EllipArc, Line, Point, Quad, Scalar, Segment, SurfaceMut,
    SurfaceOwned, Transform, EPSILON,
};
use std::{
    fmt,
    io::{Read, Write},
    str::FromStr,
    usize,
};

/// flatness of 0.05px gives good accuracy tradeoff
pub const DEFAULT_FLATNESS: Scalar = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum LineJoin {
    Miter(Scalar),
    Bevel,
    Round,
}

impl Default for LineJoin {
    fn default() -> Self {
        Self::Miter(4.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum LineCap {
    Butt,
    Square,
    Round,
}

impl Default for LineCap {
    fn default() -> Self {
        Self::Butt
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct StrokeStyle {
    pub width: Scalar,
    pub line_join: LineJoin,
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

    pub fn closed(&self) -> bool {
        self.closed
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub fn first(&self) -> Segment {
        *self.segments.first().expect("SubPath is never emtpy")
    }

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

    pub fn start(&self) -> Point {
        self.first().start()
    }

    pub fn end(&self) -> Point {
        self.last().end()
    }

    pub fn bbox(&self, init: Option<BBox>, tr: Transform) -> BBox {
        self.segments
            .iter()
            .fold(init, |bbox, seg| Some(seg.transform(tr).bbox(bbox)))
            .expect("SubPath is never empty")
    }

    pub fn reverse(&self) -> Self {
        Self {
            segments: self.segments.iter().rev().map(|s| s.reverse()).collect(),
            closed: self.closed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

/// Collection of the SubPath treated as a signle unit
#[derive(Clone, PartialEq)]
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
    /// Resource usefull for debugging: https://yqnn.github.io/svg-path-editor/
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
            if subpath.closed() {
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
            if subpath.closed() {
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

    /// Rreverse order and direction of all segments
    pub fn reverse(&self) -> Self {
        Self {
            subpaths: self.subpaths.iter().map(|s| s.reverse()).collect(),
        }
    }

    /// Rasterize mast for the path in into a provided surface.
    ///
    /// Everything that is outside of the surface will be cropped. Surface is assumed
    /// to contain zeros.
    pub fn rasterize_to<S: SurfaceMut<Item = Scalar>>(
        &self,
        tr: Transform,
        fill_rule: FillRule,
        mut surf: S,
    ) -> S {
        for line in self.flatten(tr, DEFAULT_FLATNESS, true) {
            signed_difference_line(&mut surf, line);
        }
        signed_difference_to_mask(&mut surf, fill_rule);
        surf
    }

    /// Rasterize fitted mask for the path into a provided sruface.
    ///
    /// Path is rescaled and centered appropriately to fit into a provided surface.
    pub fn rasterize_fit<S: SurfaceMut<Item = Scalar>>(
        &self,
        tr: Transform,
        fill_rule: FillRule,
        align: Align,
        surf: S,
    ) -> S {
        if surf.height() < 3 || surf.height() < 3 {
            return surf;
        }
        let src_bbox = match self.bbox(tr) {
            Some(bbox) if bbox.width() > 0.0 && bbox.height() > 0.0 => bbox,
            _ => return surf,
        };
        let dst_bbox = BBox::new(
            Point::new(1.0, 1.0),
            Point::new((surf.width() - 1) as Scalar, (surf.height() - 1) as Scalar),
        );
        let tr = Transform::fit(src_bbox, dst_bbox, align) * tr;
        self.rasterize_to(tr, fill_rule, surf)
    }

    /// Rasteraize mask for the path into an allocated surface.
    ///
    /// Surface of required size will be allocated.
    pub fn rasterize(&self, tr: Transform, fill_rule: FillRule) -> SurfaceOwned<Scalar> {
        let bbox = match self.bbox(tr) {
            Some(bbox) => bbox,
            None => return SurfaceOwned::new(0, 0),
        };
        // one pixel border to account for anti-aliasing
        let width = (bbox.width() + 2.0).ceil() as usize;
        let height = (bbox.height() + 2.0).ceil() as usize;
        let surf = SurfaceOwned::new(height, width);
        let shift = Transform::default().translate(1.0 - bbox.x(), 1.0 - bbox.y());
        self.rasterize_to(shift * tr, fill_rule, surf)
    }

    /// Save path in SVG path format.
    pub fn save(&self, mut out: impl Write) -> std::io::Result<()> {
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
            if subpath.closed() {
                out.write_all(b"Z")?;
            }
        }
        Ok(())
    }

    /// Convert path to SVG path representation
    pub fn to_svg_path(&self) -> String {
        let mut output = Vec::new();
        self.save(&mut output).expect("failed in memory write");
        String::from_utf8(output).expect("path save internal error")
    }

    /// Load path from SVG path representation
    pub fn load(mut input: impl Read) -> std::io::Result<Self> {
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer)?;
        let parser = PathParser::new(&buffer);
        let mut builder = PathBuilder::new();
        parser.parse(&mut builder)?;
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
    subpath: usize,
    segment: usize,
    stack: Vec<Result<Cubic, Quad>>,
}

impl<'a> PathFlattenIter<'a> {
    fn new(path: &'a Path, transform: Transform, flatness: Scalar, close: bool) -> Self {
        Self {
            path,
            transform,
            flatness: 16.0 * flatness * flatness,
            close,
            subpath: 0,
            segment: 0,
            stack: Default::default(),
        }
    }
}

impl<'a> Iterator for PathFlattenIter<'a> {
    type Item = Line;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.stack.pop() {
                Some(Ok(cubic)) => {
                    if cubic.flatness() < self.flatness {
                        return Some(Line::new(cubic.start(), cubic.end()));
                    }
                    let (c0, c1) = cubic.split();
                    self.stack.push(Ok(c1));
                    self.stack.push(Ok(c0));
                }
                Some(Err(quad)) => {
                    if quad.flatness() < self.flatness {
                        return Some(Line::new(quad.start(), quad.end()));
                    }
                    let (q0, q1) = quad.split();
                    self.stack.push(Err(q1));
                    self.stack.push(Err(q0));
                }
                None => {
                    let subpath = self.path.subpaths.get(self.subpath)?;
                    match subpath.segments().get(self.segment) {
                        None => {
                            self.subpath += 1;
                            self.segment = 0;
                            if subpath.closed || self.close {
                                let line = Line::new(subpath.end(), subpath.start())
                                    .transform(self.transform);
                                return Some(line);
                            }
                        }
                        Some(segment) => {
                            self.segment += 1;
                            match segment {
                                Segment::Line(line) => return Some(line.transform(self.transform)),
                                Segment::Quad(quad) => {
                                    self.stack.push(Err(quad.transform(self.transform)));
                                }
                                Segment::Cubic(cubic) => {
                                    self.stack.push(Ok(cubic.transform(self.transform)));
                                }
                            }
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
        } = std::mem::replace(self, Default::default());
        subpaths.extend(SubPath::new(subpath, false));
        Path::new(subpaths)
    }

    /// Extend path from string, which is specified in the same format as SVGs path element.
    pub fn append_svg_path(&mut self, string: impl AsRef<[u8]>) -> Result<&mut Self, Error> {
        let parser = PathParser::new(string.as_ref());
        parser.parse(self)?;
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
        let Point([rx, ry]) = radii.into();
        let rx = rx.abs();
        let ry = ry.abs();
        let radii = Point::new(rx, ry);

        let size = size.into();
        let lx = self.position.x();
        let ly = self.position.y();
        let hx = lx + size.x().abs();
        let hy = ly + size.y().abs();
        let rounded = rx > EPSILON && ry > EPSILON;

        if 2.0 * rx > hx - lx || 2.0 * ry > hy - ly {
            return self;
        }

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
        self.close().move_to(Point::new(lx, ly))
    }

    /// Current possition of the builder
    pub fn position(&self) -> Point {
        self.position
    }
}

impl FromStr for Path {
    type Err = Error;

    fn from_str(text: &str) -> Result<Path, Self::Err> {
        let mut builder = PathBuilder::new();
        let parser = PathParser::new(text.as_ref());
        parser.parse(&mut builder)?;
        Ok(builder.build())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Error {
    ParseError { reason: String, offset: usize },
    ConvertionError { reason: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<Error> for std::io::Error {
    fn from(error: Error) -> Self {
        Self::new(std::io::ErrorKind::InvalidData, error)
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub struct PathParser<'a> {
    // text containing unparsed path
    text: &'a [u8],
    // current offset in the text
    offset: usize,
    // previous command
    prev_cmd: Option<u8>,
    // current position from which next curve will start
    position: Point,
}

impl<'a> PathParser<'a> {
    fn new(text: &'a [u8]) -> PathParser<'a> {
        Self {
            text,
            offset: 0,
            prev_cmd: None,
            position: Point::new(0.0, 0.0),
        }
    }

    /// Error construction helper
    fn error<S: Into<String>>(&self, reason: S) -> Error {
        Error::ParseError {
            offset: self.offset,
            reason: reason.into(),
        }
    }

    /// Byte at the current position
    fn current(&self) -> Result<u8, Error> {
        match self.text.get(self.offset) {
            Some(byte) => Ok(*byte),
            None => Err(self.error("unexpected end of input")),
        }
    }

    /// Advance current position by `count` bytes
    fn advance(&mut self, count: usize) {
        self.offset += count;
    }

    /// Check if end of file is reached
    fn is_eof(&self) -> bool {
        self.offset >= self.text.len()
    }

    /// Consume insignificant separators
    fn parse_separators(&mut self) {
        while !self.is_eof() {
            match self.text[self.offset] {
                b' ' | b'\t' | b'\r' | b'\n' | b',' => {
                    self.offset += 1;
                }
                _ => break,
            }
        }
    }

    /// Check if byte under the cursor is a digit and advance
    fn parse_digits(&mut self) -> Result<bool, Error> {
        let mut found = false;
        loop {
            match self.current() {
                Ok(b'0'..=b'9') => {
                    self.advance(1);
                    found = true;
                }
                _ => return Ok(found),
            }
        }
    }

    /// Consume `+|-` sign
    fn parse_sign(&mut self) -> Result<(), Error> {
        match self.current()? {
            b'-' | b'+' => {
                self.advance(1);
            }
            _ => (),
        }
        Ok(())
    }

    /// Parse single scalar
    fn parse_scalar(&mut self) -> Result<Scalar, Error> {
        self.parse_separators();
        let start = self.offset;
        self.parse_sign()?;
        let whole = self.parse_digits()?;
        if !self.is_eof() {
            let fraction = match self.current()? {
                b'.' => {
                    self.advance(1);
                    self.parse_digits()?
                }
                _ => false,
            };
            if !whole && !fraction {
                return Err(self.error("failed to parse scalar"));
            }
            match self.current()? {
                b'e' | b'E' => {
                    self.advance(1);
                    self.parse_sign()?;
                    if !self.parse_digits()? {
                        return Err(self.error("failed to parse scalar"));
                    }
                }
                _ => (),
            }
        }
        // unwrap is safe here since we have validated content
        let scalar_str = std::str::from_utf8(&self.text[start..self.offset]).unwrap();
        let scalar = Scalar::from_str(scalar_str).unwrap();
        Ok(scalar)
    }

    /// Parse pair of scalars and convert it to a point
    fn parse_point(&mut self) -> Result<Point, Error> {
        let x = self.parse_scalar()?;
        let y = self.parse_scalar()?;
        let is_relative = match self.prev_cmd {
            Some(cmd) => cmd.is_ascii_lowercase(),
            None => false,
        };
        if is_relative {
            Ok(Point([x, y]) + self.position)
        } else {
            Ok(Point([x, y]))
        }
    }

    /// Parse SVG flag `0|1` used by elliptic arc command
    fn parse_flag(&mut self) -> Result<bool, Error> {
        self.parse_separators();
        match self.current()? {
            b'0' => {
                self.advance(1);
                Ok(false)
            }
            b'1' => {
                self.advance(1);
                Ok(true)
            }
            _ => Err(self.error("failed to parse flag")),
        }
    }

    /// Parse SVG command
    fn parse_cmd(&mut self) -> Result<u8, Error> {
        let cmd = self.current()?;
        match cmd {
            b'M' | b'm' | b'L' | b'l' | b'V' | b'v' | b'H' | b'h' | b'C' | b'c' | b'S' | b's'
            | b'Q' | b'q' | b'T' | b't' | b'A' | b'a' | b'Z' | b'z' => {
                self.advance(1);
                self.prev_cmd = if cmd == b'm' {
                    Some(b'l')
                } else if cmd == b'M' {
                    Some(b'L')
                } else if cmd == b'Z' || cmd == b'z' {
                    None
                } else {
                    Some(cmd)
                };
                Ok(cmd)
            }
            _ => match self.prev_cmd {
                Some(cmd) => Ok(cmd),
                None => Err(self.error("failed to parse path cmd")),
            },
        }
    }

    /// Parse SVG path and apply changes to the path builder.
    fn parse(mut self, builder: &mut PathBuilder) -> Result<(), Error> {
        loop {
            self.parse_separators();
            if self.is_eof() {
                break;
            }
            self.position = builder.position();
            let cmd = self.parse_cmd()?;
            match cmd {
                b'M' | b'm' => {
                    builder.move_to(self.parse_point()?);
                }
                b'L' | b'l' => {
                    builder.line_to(self.parse_point()?);
                }
                b'V' | b'v' => {
                    let y = self.parse_scalar()?;
                    let p0 = builder.position();
                    let p1 = if cmd == b'v' {
                        Point::new(p0.x(), p0.y() + y)
                    } else {
                        Point::new(p0.x(), y)
                    };
                    builder.line_to(p1);
                }
                b'H' | b'h' => {
                    let x = self.parse_scalar()?;
                    let p0 = builder.position();
                    let p1 = if cmd == b'h' {
                        Point::new(p0.x() + x, p0.y())
                    } else {
                        Point::new(x, p0.y())
                    };
                    builder.line_to(p1);
                }
                b'Q' | b'q' => {
                    builder.quad_to(self.parse_point()?, self.parse_point()?);
                }
                b'T' | b't' => {
                    builder.quad_smooth_to(self.parse_point()?);
                }
                b'C' | b'c' => {
                    builder.cubic_to(
                        self.parse_point()?,
                        self.parse_point()?,
                        self.parse_point()?,
                    );
                }
                b'S' | b's' => {
                    builder.cubic_smooth_to(self.parse_point()?, self.parse_point()?);
                }
                b'A' | b'a' => {
                    let rx = self.parse_scalar()?;
                    let ry = self.parse_scalar()?;
                    let x_axis_rot = self.parse_scalar()?;
                    let large_flag = self.parse_flag()?;
                    let sweep_flag = self.parse_flag()?;
                    let dst = self.parse_point()?;
                    builder.arc_to((rx, ry), x_axis_rot, large_flag, sweep_flag, dst);
                }
                b'Z' | b'z' => {
                    builder.close();
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq, Surface, PI};

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
    fn test_path_parse() -> Result<(), Error> {
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
        assert_eq!(format!("{:?}", path), format!("{:?}", reference));

        let path: Path = " M0,0L1-1L1,0ZL0,1 L1,1Z ".parse()?;
        let reference = Path::new(vec![
            SubPath::new(
                vec![
                    Line::new((0.0, 0.0), (1.0, -1.0)).into(),
                    Line::new((1.0, -1.0), (1.0, 0.0)).into(),
                ],
                true,
            )
            .unwrap(),
            SubPath::new(
                vec![
                    Line::new((0.0, 0.0), (0.0, 1.0)).into(),
                    Line::new((0.0, 1.0), (1.0, 1.0)).into(),
                ],
                true,
            )
            .unwrap(),
        ]);
        assert_eq!(format!("{:?}", path), format!("{:?}", reference));
        Ok(())
    }

    #[test]
    fn test_save_load() -> std::io::Result<()> {
        let path: Path = SQUIRREL.parse()?;
        let mut path_save = Vec::new();
        path.save(&mut path_save)?;
        let path_load = Path::load(std::io::Cursor::new(path_save))?;
        assert_eq!(format!("{:?}", path), format!("{:?}", path_load));
        Ok(())
    }

    #[test]
    fn test_flatten() -> Result<(), Error> {
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

    #[test]
    fn test_fill_rule() -> Result<(), Error> {
        let tr = Transform::default();
        let path: Path = r#"
            M50,0 21,90 98,35 2,35 79,90z
            M110,0 h90 v90 h-90z
            M130,20 h50 v50 h-50 z
            M210,0  h90 v90 h-90 z
            M230,20 v50 h50 v-50 z
        "#
        .parse()?;
        let y = 50;
        let x0 = 50; // middle of the star
        let x1 = 150; // middle of the first box
        let x2 = 250; // middle of the second box

        let surf = path.rasterize(tr, FillRule::EvenOdd);
        assert_approx_eq!(surf.get(y, x0).unwrap(), 0.0);
        assert_approx_eq!(surf.get(y, x1).unwrap(), 0.0);
        assert_approx_eq!(surf.get(y, x2).unwrap(), 0.0);
        let area = surf.iter().sum::<Scalar>();
        assert_approx_eq!(area, 13130.0, 1.0);

        let surf = path.rasterize(tr, FillRule::NonZero);
        assert_approx_eq!(surf.get(y, x0).unwrap(), 1.0);
        assert_approx_eq!(surf.get(y, x1).unwrap(), 1.0);
        assert_approx_eq!(surf.get(y, x2).unwrap(), 0.0);
        let area = surf.iter().sum::<Scalar>();
        assert_approx_eq!(area, 16492.5, 1.0);

        Ok(())
    }
}
