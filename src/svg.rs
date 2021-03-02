//! SVG path parser
//!
//! See [SVG Path Specification](https://www.w3.org/TR/SVG11/paths.html#PathData)
use crate::{PathBuilder, Point, Scalar};
use std::{fmt, io::Read};

/// Possible SVG path commands
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SVGPathCmd {
    MoveTo(Point),
    LineTo(Point),
    QuadTo(Point, Point),
    CubicTo(Point, Point, Point),
    ArcTo {
        radii: Point,
        x_axis_rot: Scalar,
        large: bool,
        sweep: bool,
        dst: Point,
    },
    Close(Point),
}

impl SVGPathCmd {
    /// Get destination point of the SVG command
    pub fn dst(&self) -> Point {
        use SVGPathCmd::*;
        *match self {
            MoveTo(dst) => dst,
            LineTo(dst) => dst,
            QuadTo(_, dst) => dst,
            CubicTo(_, _, dst) => dst,
            ArcTo { dst, .. } => dst,
            Close(dst) => dst,
        }
    }

    /// Apply SVG command to path builder
    pub fn apply(&self, builder: &mut PathBuilder) {
        use SVGPathCmd::*;
        match self {
            MoveTo(p) => builder.move_to(p),
            LineTo(p) => builder.line_to(p),
            QuadTo(p1, p2) => builder.quad_to(p1, p2),
            CubicTo(p1, p2, p3) => builder.cubic_to(p1, p2, p3),
            Close(_) => builder.close(),
            ArcTo {
                radii,
                x_axis_rot,
                large,
                sweep,
                dst,
            } => builder.arc_to(radii, *x_axis_rot, *large, *sweep, dst),
        };
    }
}

/// Path parser for SVG encoded path
///
/// See [SVG Path Specification](https://www.w3.org/TR/SVG11/paths.html#PathData)
#[derive(Debug)]
pub struct SVGPathParser<I> {
    // input containing unparsed SVG path
    input: I,
    // read but not consumed input
    input_buffer: Option<u8>,
    // parser buffer
    parse_buffer: Vec<u8>,
    // previous operation
    prev_op: Option<u8>,
    // previous command (used to determine smooth points)
    prev_cmd: Option<SVGPathCmd>,
    // current position from which next relative curve will start
    position: Point,
    // current sub-path starting position
    subpath_start: Point,
}

impl<I: Read> SVGPathParser<I> {
    pub fn new(input: I) -> Self {
        Self {
            input,
            input_buffer: None,
            parse_buffer: Default::default(),
            prev_op: None,
            prev_cmd: None,
            position: Point::new(0.0, 0.0),
            subpath_start: Point::new(0.0, 0.0),
        }
    }

    // consume single byte from the input
    fn parse_byte(&mut self) -> Result<Option<u8>, SVGPathParserError> {
        match self.input_buffer.take() {
            None => {
                let mut byte = [0; 1];
                if self.input.read(&mut byte)? != 0 {
                    Ok(Some(byte[0]))
                } else {
                    Ok(None)
                }
            }
            byte => Ok(byte),
        }
    }

    fn unparse_byte(&mut self, byte: u8) {
        debug_assert!(self.input_buffer.is_none());
        self.input_buffer = Some(byte);
    }

    // consume input while `pred` predicate is true, consumed input is stored in `Self::buffer`
    fn parse_while(
        &mut self,
        mut pred: impl FnMut(u8) -> bool,
    ) -> Result<usize, SVGPathParserError> {
        let mut count = 0;
        loop {
            let byte = match self.parse_byte()? {
                None => break,
                Some(byte) => byte,
            };
            if !pred(byte) {
                self.unparse_byte(byte);
                break;
            }
            count += 1;
            self.parse_buffer.push(byte);
        }
        Ok(count)
    }

    // consume at most one byte from the input, if predicate returns true
    fn parse_once(&mut self, pred: impl FnOnce(u8) -> bool) -> Result<bool, SVGPathParserError> {
        let byte = match self.parse_byte()? {
            None => return Ok(false),
            Some(byte) => byte,
        };
        if pred(byte) {
            self.parse_buffer.push(byte);
            Ok(true)
        } else {
            self.unparse_byte(byte);
            Ok(false)
        }
    }

    // consume seprators from the input
    fn parse_separators(&mut self) -> Result<(), SVGPathParserError> {
        loop {
            let byte = match self.parse_byte()? {
                None => break,
                Some(byte) => byte,
            };
            if !matches!(byte, b' ' | b'\t' | b'\r' | b'\n' | b',') {
                self.unparse_byte(byte);
                break;
            }
        }
        Ok(())
    }

    // parse single scalar value from the input
    fn parse_scalar(&mut self) -> Result<Scalar, SVGPathParserError> {
        self.parse_separators()?;

        self.parse_buffer.clear();
        self.parse_once(|byte| matches!(byte, b'-' | b'+'))?;
        let whole = self.parse_while(|byte| matches!(byte, b'0'..=b'9'))?;
        let frac = if self.parse_once(|byte| matches!(byte, b'.'))? {
            self.parse_while(|byte| matches!(byte, b'0'..=b'9'))?
        } else {
            0
        };
        if whole + frac == 0 {
            return Err(SVGPathParserError::InvalidScalar);
        }
        if self.parse_once(|byte| matches!(byte, b'e' | b'E'))? {
            self.parse_once(|byte| matches!(byte, b'-' | b'+'))?;
            if self.parse_while(|byte| matches!(byte, b'0'..=b'9'))? == 0 {
                return Err(SVGPathParserError::InvalidScalar);
            }
        }

        // unwrap is safe here since we have validated content
        let scalar_str = std::str::from_utf8(self.parse_buffer.as_ref()).unwrap();
        let scalar = scalar_str.parse().unwrap();

        self.parse_buffer.clear();
        Ok(scalar)
    }

    // parse pair of scalars and convert it to a point
    fn parse_point(&mut self) -> Result<Point, SVGPathParserError> {
        let x = self.parse_scalar()?;
        let y = self.parse_scalar()?;
        let is_relative = match self.prev_op {
            Some(cmd) => cmd.is_ascii_lowercase(),
            None => false,
        };
        if is_relative {
            Ok(Point([x, y]) + self.position)
        } else {
            Ok(Point([x, y]))
        }
    }

    // parse flag `0|1` used by elliptic arc command
    fn parse_flag(&mut self) -> Result<bool, SVGPathParserError> {
        self.parse_separators()?;
        match self.parse_byte()? {
            Some(b'0') => Ok(false),
            Some(b'1') => Ok(true),
            byte => {
                if let Some(byte) = byte {
                    self.unparse_byte(byte);
                }
                Err(SVGPathParserError::InvalidFlag)
            }
        }
    }

    // parse svg command, none indicates end of input
    fn parse_op(&mut self) -> Result<Option<u8>, SVGPathParserError> {
        let op = match self.parse_byte()? {
            None => return Ok(None),
            Some(op) => op,
        };
        match op {
            b'M' | b'm' | b'L' | b'l' | b'V' | b'v' | b'H' | b'h' | b'C' | b'c' | b'S' | b's'
            | b'Q' | b'q' | b'T' | b't' | b'A' | b'a' | b'Z' | b'z' => {
                self.prev_op = if op == b'm' {
                    Some(b'l')
                } else if op == b'M' {
                    Some(b'L')
                } else if op == b'Z' || op == b'z' {
                    None
                } else {
                    Some(op)
                };
                Ok(Some(op))
            }
            byte => {
                self.unparse_byte(byte);
                match self.prev_op {
                    Some(op) => Ok(Some(op)),
                    None => Err(SVGPathParserError::InvalidCmd(op)),
                }
            }
        }
    }

    /// Parse single SVG path command from the input
    pub fn parse_cmd(&mut self) -> Result<Option<SVGPathCmd>, SVGPathParserError> {
        self.parse_separators()?;
        let op = match self.parse_op()? {
            None => return Ok(None),
            Some(op) => op,
        };
        let cmd = match op {
            b'M' | b'm' => {
                let dst = self.parse_point()?;
                self.subpath_start = dst;
                SVGPathCmd::MoveTo(dst)
            }
            b'L' | b'l' => SVGPathCmd::LineTo(self.parse_point()?),
            b'V' | b'v' => {
                let y = self.parse_scalar()?;
                let p0 = self.position;
                let p1 = if op == b'v' {
                    Point::new(p0.x(), p0.y() + y)
                } else {
                    Point::new(p0.x(), y)
                };
                SVGPathCmd::LineTo(p1)
            }
            b'H' | b'h' => {
                let x = self.parse_scalar()?;
                let p0 = self.position;
                let p1 = if op == b'h' {
                    Point::new(p0.x() + x, p0.y())
                } else {
                    Point::new(x, p0.y())
                };
                SVGPathCmd::LineTo(p1)
            }
            b'Q' | b'q' => SVGPathCmd::QuadTo(self.parse_point()?, self.parse_point()?),
            b'T' | b't' => {
                let p1 = match self.prev_cmd {
                    Some(SVGPathCmd::QuadTo(p1, p2)) => 2.0 * p2 - p1,
                    _ => self.position,
                };
                let p2 = self.parse_point()?;
                SVGPathCmd::QuadTo(p1, p2)
            }
            b'C' | b'c' => SVGPathCmd::CubicTo(
                self.parse_point()?,
                self.parse_point()?,
                self.parse_point()?,
            ),
            b'S' | b's' => {
                let p1 = match self.prev_cmd {
                    Some(SVGPathCmd::CubicTo(_, p2, p3)) => 2.0 * p3 - p2,
                    _ => self.position,
                };
                let p2 = self.parse_point()?;
                let p3 = self.parse_point()?;
                SVGPathCmd::CubicTo(p1, p2, p3)
            }
            b'A' | b'a' => {
                let rx = self.parse_scalar()?;
                let ry = self.parse_scalar()?;
                let x_axis_rot = self.parse_scalar()?;
                let large_flag = self.parse_flag()?;
                let sweep_flag = self.parse_flag()?;
                let dst = self.parse_point()?;
                SVGPathCmd::ArcTo {
                    radii: Point::new(rx, ry),
                    x_axis_rot,
                    large: large_flag,
                    sweep: sweep_flag,
                    dst,
                }
            }
            b'Z' | b'z' => SVGPathCmd::Close(self.subpath_start),
            _ => unreachable!(),
        };
        self.position = cmd.dst();
        self.prev_cmd = Some(cmd);
        Ok(self.prev_cmd)
    }
}

impl<I: Read> Iterator for SVGPathParser<I> {
    type Item = Result<SVGPathCmd, SVGPathParserError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.parse_cmd().transpose()
    }
}

#[derive(Debug)]
pub enum SVGPathParserError {
    InvalidCmd(u8),
    InvalidScalar,
    InvalidFlag,
    UnexpectedSegmentType,
    IOError(std::io::Error),
}

impl fmt::Display for SVGPathParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<std::io::Error> for SVGPathParserError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

impl From<SVGPathParserError> for std::io::Error {
    fn from(error: SVGPathParserError) -> Self {
        match error {
            SVGPathParserError::IOError(error) => error,
            _ => Self::new(std::io::ErrorKind::InvalidData, error),
        }
    }
}

impl std::error::Error for SVGPathParserError {}
