//! SVG path parser
//!
//! See [SVG Path Specification](https://www.w3.org/TR/SVG11/paths.html#PathData)
use crate::{PathBuilder, Point, Scalar, Transform};
use std::{
    fmt,
    io::{Cursor, Read},
    str::FromStr,
};

/// Possible SVG path commands
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SvgPathCmd {
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

impl SvgPathCmd {
    /// Get destination point of the SVG command
    pub fn dst(&self) -> Point {
        use SvgPathCmd::*;
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
        use SvgPathCmd::*;
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

struct Parser<I> {
    input: I,
    input_buffer: Option<u8>,
}

impl<I: Read> Parser<I> {
    pub fn new(input: I) -> Self {
        Self {
            input,
            input_buffer: None,
        }
    }

    // consume single byte from the input
    pub fn parse_byte(&mut self) -> Result<Option<u8>, SvgParserError> {
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

    // put byte into input buffer, at most one byte is cached
    pub fn unparse_byte(&mut self, byte: u8) {
        debug_assert!(self.input_buffer.is_none());
        self.input_buffer = Some(byte);
    }

    // consume input while `pred` predicate is true, consumed input is stored in `Self::buffer`
    pub fn parse_while(
        &mut self,
        mut pred: impl FnMut(u8) -> bool,
        mut proc: impl FnMut(u8),
    ) -> Result<usize, SvgParserError> {
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
            proc(byte);
        }
        Ok(count)
    }

    // consume at most one byte from the input, if predicate returns true
    pub fn parse_once(
        &mut self,
        pred: impl FnOnce(u8) -> bool,
        proc: impl FnOnce(u8),
    ) -> Result<bool, SvgParserError> {
        let byte = match self.parse_byte()? {
            None => return Ok(false),
            Some(byte) => byte,
        };
        if pred(byte) {
            proc(byte);
            Ok(true)
        } else {
            self.unparse_byte(byte);
            Ok(false)
        }
    }

    // consume separators from the input
    pub fn parse_separators(&mut self) -> Result<(), SvgParserError> {
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
    pub fn parse_scalar(&mut self) -> Result<Scalar, SvgParserError> {
        self.parse_separators()?;

        let mut mantissa: i64 = 0;
        let mut exponent: i64 = 0;
        let mut sign = 1;

        fn push_digit(value: &mut i64, byte: u8) {
            let digit = byte - b'0';
            *value = value.wrapping_mul(10).wrapping_add(digit as i64);
        }

        self.parse_once(
            |byte| matches!(byte, b'-' | b'+'),
            |byte| {
                if byte == b'-' {
                    sign = -1
                }
            },
        )?;
        let whole = self.parse_while(
            |byte| byte.is_ascii_digit(),
            |byte| push_digit(&mut mantissa, byte),
        )?;
        let matches_dot = self.parse_once(|byte| matches!(byte, b'.'), |_| {})?;
        let frac = if matches_dot {
            self.parse_while(
                |byte| byte.is_ascii_digit(),
                |byte| {
                    push_digit(&mut mantissa, byte);
                    exponent -= 1;
                },
            )?
        } else {
            0
        };
        mantissa *= sign;

        if whole + frac == 0 {
            return Err(SvgParserError::InvalidScalar);
        }

        let matches_exp = self.parse_once(|byte| matches!(byte, b'e' | b'E'), |_| {})?;
        if matches_exp {
            let mut sci: i64 = 0;
            let mut sci_sign = 1;
            self.parse_once(
                |byte| matches!(byte, b'-' | b'+'),
                |byte| {
                    if byte == b'-' {
                        sci_sign = -1
                    }
                },
            )?;
            if self.parse_while(
                |byte| byte.is_ascii_digit(),
                |byte| push_digit(&mut sci, byte),
            )? == 0
            {
                return Err(SvgParserError::InvalidScalar);
            }
            exponent = exponent.wrapping_add(sci_sign * sci)
        }

        let ten: Scalar = 10.0;
        Ok((mantissa as Scalar) * ten.powi(exponent as i32))
    }
}

/// Path parser for SVG encoded path
///
/// See [SVG Path Specification](https://www.w3.org/TR/SVG11/paths.html#PathData)
pub struct SvgPathParser<I> {
    parser: Parser<I>,
    // previous operation
    prev_op: Option<u8>,
    // previous command (used to determine smooth points)
    prev_cmd: Option<SvgPathCmd>,
    // current position from which next relative curve will start
    position: Point,
    // current sub-path starting position
    subpath_start: Point,
}

impl<I: Read> SvgPathParser<I> {
    pub fn new(input: I) -> Self {
        Self {
            parser: Parser::new(input),
            prev_op: None,
            prev_cmd: None,
            position: Point::new(0.0, 0.0),
            subpath_start: Point::new(0.0, 0.0),
        }
    }

    // parse pair of scalars and convert it to a point
    fn parse_point(&mut self) -> Result<Point, SvgParserError> {
        let point = Point::new(self.parser.parse_scalar()?, self.parser.parse_scalar()?);
        match self.prev_op {
            Some(cmd) if cmd.is_ascii_lowercase() => Ok(point + self.position),
            _ => Ok(point),
        }
    }

    // parse flag `0|1` used by elliptic arc command
    fn parse_flag(&mut self) -> Result<bool, SvgParserError> {
        self.parser.parse_separators()?;
        match self.parser.parse_byte()? {
            Some(b'0') => Ok(false),
            Some(b'1') => Ok(true),
            byte => {
                if let Some(byte) = byte {
                    self.parser.unparse_byte(byte);
                }
                Err(SvgParserError::InvalidFlag)
            }
        }
    }

    // parse svg command, none indicates end of input
    fn parse_op(&mut self) -> Result<Option<u8>, SvgParserError> {
        let op = match self.parser.parse_byte()? {
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
                self.parser.unparse_byte(byte);
                match self.prev_op {
                    Some(op) => Ok(Some(op)),
                    None => Err(SvgParserError::InvalidCmd(op)),
                }
            }
        }
    }

    /// Parse single SVG path command from the input
    pub fn parse_cmd(&mut self) -> Result<Option<SvgPathCmd>, SvgParserError> {
        self.parser.parse_separators()?;
        let op = match self.parse_op()? {
            None => return Ok(None),
            Some(op) => op,
        };
        let cmd = match op {
            b'M' | b'm' => {
                let dst = self.parse_point()?;
                self.subpath_start = dst;
                SvgPathCmd::MoveTo(dst)
            }
            b'L' | b'l' => SvgPathCmd::LineTo(self.parse_point()?),
            b'V' | b'v' => {
                let y = self.parser.parse_scalar()?;
                let p0 = self.position;
                let p1 = if op == b'v' {
                    Point::new(p0.x(), p0.y() + y)
                } else {
                    Point::new(p0.x(), y)
                };
                SvgPathCmd::LineTo(p1)
            }
            b'H' | b'h' => {
                let x = self.parser.parse_scalar()?;
                let p0 = self.position;
                let p1 = if op == b'h' {
                    Point::new(p0.x() + x, p0.y())
                } else {
                    Point::new(x, p0.y())
                };
                SvgPathCmd::LineTo(p1)
            }
            b'Q' | b'q' => SvgPathCmd::QuadTo(self.parse_point()?, self.parse_point()?),
            b'T' | b't' => {
                let p1 = match self.prev_cmd {
                    Some(SvgPathCmd::QuadTo(p1, p2)) => 2.0 * p2 - p1,
                    _ => self.position,
                };
                let p2 = self.parse_point()?;
                SvgPathCmd::QuadTo(p1, p2)
            }
            b'C' | b'c' => SvgPathCmd::CubicTo(
                self.parse_point()?,
                self.parse_point()?,
                self.parse_point()?,
            ),
            b'S' | b's' => {
                let p1 = match self.prev_cmd {
                    Some(SvgPathCmd::CubicTo(_, p2, p3)) => 2.0 * p3 - p2,
                    _ => self.position,
                };
                let p2 = self.parse_point()?;
                let p3 = self.parse_point()?;
                SvgPathCmd::CubicTo(p1, p2, p3)
            }
            b'A' | b'a' => {
                let rx = self.parser.parse_scalar()?;
                let ry = self.parser.parse_scalar()?;
                let x_axis_rot = self.parser.parse_scalar()?;
                let large_flag = self.parse_flag()?;
                let sweep_flag = self.parse_flag()?;
                let dst = self.parse_point()?;
                SvgPathCmd::ArcTo {
                    radii: Point::new(rx, ry),
                    x_axis_rot,
                    large: large_flag,
                    sweep: sweep_flag,
                    dst,
                }
            }
            b'Z' | b'z' => SvgPathCmd::Close(self.subpath_start),
            _ => unreachable!(),
        };
        self.position = cmd.dst();
        self.prev_cmd = Some(cmd);
        Ok(self.prev_cmd)
    }
}

impl<I: Read> Iterator for SvgPathParser<I> {
    type Item = Result<SvgPathCmd, SvgParserError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.parse_cmd().transpose()
    }
}

// maximum length of matrix | translate(X|Y)? | scale(X|Y)? | rotate | skew(X|Y) | deg | rad
const TRANSFORM_BUF: usize = 10;

struct SvgTransformParser<I> {
    parser: Parser<I>,
    buf: [u8; TRANSFORM_BUF],
    buf_len: usize,
}

impl<I: Read> SvgTransformParser<I> {
    fn new(input: I) -> Self {
        Self {
            parser: Parser::new(input),
            buf: [0; TRANSFORM_BUF],
            buf_len: 0,
        }
    }

    fn parse_ident(&mut self) -> Result<&[u8], SvgParserError> {
        self.buf_len = 0;
        self.parser.parse_while(
            |b| b.is_ascii_alphabetic(),
            |b| {
                self.buf[self.buf_len] = b;
                self.buf_len = (self.buf_len + 1) % TRANSFORM_BUF;
            },
        )?;

        Ok(&self.buf[..self.buf_len])
    }

    // parse angle in radians
    fn parse_angle(&mut self) -> Result<Scalar, SvgParserError> {
        let value = self.parser.parse_scalar()?;
        let units = self.parse_ident()?;
        match units {
            b"" | b"deg" => Ok(value * crate::PI / 180.0),
            b"rad" => Ok(value),
            _ => Err(SvgParserError::InvalidUnits),
        }
    }

    fn parse_length(&mut self) -> Result<Scalar, SvgParserError> {
        let value = self.parser.parse_scalar()?;
        let _units = self.parse_ident()?;
        // TODO: units px|cm|in? ...
        Ok(value)
    }

    fn parse_transform(&mut self) -> Result<Option<Transform>, SvgParserError> {
        enum Op {
            Matrix,
            Rotate,
            Translate,
            TranslateX,
            TranslateY,
            Scale,
            ScaleX,
            ScaleY,
            SkewX,
            SkewY,
        }

        self.parser.parse_separators()?;
        match self.parser.parse_byte()? {
            None => return Ok(None),
            Some(b) => self.parser.unparse_byte(b),
        }

        let op = match self.parse_ident()? {
            b"matrix" => Op::Matrix,
            b"rotate" => Op::Rotate,
            b"translate" => Op::Translate,
            b"translateX" => Op::TranslateX,
            b"translateY" => Op::TranslateY,
            b"scale" => Op::Scale,
            b"scaleX" => Op::ScaleX,
            b"scaleY" => Op::ScaleY,
            b"skewX" => Op::SkewX,
            b"skewY" => Op::SkewY,
            _ => {
                return Err(SvgParserError::InvalidTransformOp);
            }
        };

        self.parser.parse_separators()?;
        if !matches!(self.parser.parse_byte()?, Some(b'(')) {
            return Err(SvgParserError::BracketExpected);
        }

        let tr = match op {
            Op::Matrix => {
                let m00 = self.parser.parse_scalar()?;
                let m10 = self.parser.parse_scalar()?;
                let m01 = self.parser.parse_scalar()?;
                let m11 = self.parser.parse_scalar()?;
                let m02 = self.parser.parse_scalar()?;
                let m12 = self.parser.parse_scalar()?;
                Transform::new(m00, m01, m02, m10, m11, m12)
            }
            Op::Rotate => {
                let mut tr = Transform::new_rotate(self.parse_angle()?);
                if let Ok(tx) = self.parse_length() {
                    let ty = self.parse_length()?;
                    // rotate around (tx, ty)
                    tr = Transform::new_translate(tx, ty)
                        .pre_concat(tr)
                        .pre_translate(-tx, -ty);
                }
                tr
            }
            Op::Translate => {
                let tx = self.parse_length()?;
                let ty = self.parse_length().unwrap_or(0.0);
                Transform::new_translate(tx, ty)
            }
            Op::TranslateX => {
                let tx = self.parse_length()?;
                Transform::new_translate(tx, 0.0)
            }
            Op::TranslateY => {
                let ty = self.parse_length()?;
                Transform::new_translate(0.0, ty)
            }
            Op::Scale => {
                let sx = self.parser.parse_scalar()?;
                let sy = self.parser.parse_scalar().unwrap_or(sx);
                Transform::new_scale(sx, sy)
            }
            Op::ScaleX => {
                let sx = self.parser.parse_scalar()?;
                Transform::new_scale(sx, 1.0)
            }
            Op::ScaleY => {
                let sy = self.parser.parse_scalar()?;
                Transform::new_scale(1.0, sy)
            }
            Op::SkewX => {
                let ax = self.parse_angle()?;
                Transform::new_skew(ax, 0.0)
            }
            Op::SkewY => {
                let ay = self.parse_angle()?;
                Transform::new_skew(0.0, ay)
            }
        };

        self.parser.parse_separators()?;
        if !matches!(self.parser.parse_byte()?, Some(b')')) {
            return Err(SvgParserError::BracketExpected);
        }

        Ok(Some(tr))
    }
}

impl FromStr for Transform {
    type Err = SvgParserError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let mut tr = Transform::identity();
        let mut parser = SvgTransformParser::new(Cursor::new(text));
        while let Some(tr_next) = parser.parse_transform()? {
            tr = tr * tr_next;
        }
        Ok(tr)
    }
}

/// Error while parsing path in the SVG format
#[derive(Debug)]
pub enum SvgParserError {
    /// Failed to parse SVG command
    InvalidCmd(u8),
    /// Failed to parse scalar value
    InvalidScalar,
    /// Failed to parse flag value
    InvalidFlag,
    /// Unexpected segment type found while parsing curve segment
    UnexpectedSegmentType,
    /// Invalid transform operation
    InvalidTransformOp,
    /// Invalid bounding box
    InvalidBBox,
    /// Invalid (Angle|Gradient|Length)
    InvalidUnits,
    /// Invalid Fill Rule
    InvalidFillRule,
    /// Bracket expected,
    BracketExpected,
    /// JSON error
    #[cfg(feature = "serde")]
    Json(serde_json::Error),
    /// IO error propagated while reading input stream
    IoError(std::io::Error),
}

impl fmt::Display for SvgParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SvgPathParser::{:?}", self)
    }
}

impl From<std::io::Error> for SvgParserError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError(error)
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for SvgParserError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<SvgParserError> for std::io::Error {
    fn from(error: SvgParserError) -> Self {
        match error {
            SvgParserError::IoError(error) => error,
            _ => Self::new(std::io::ErrorKind::InvalidData, error),
        }
    }
}

impl std::error::Error for SvgParserError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;
    use std::io::Cursor;

    #[test]
    fn test_parse_scalar() -> Result<(), SvgParserError> {
        let mut parser = Parser::new(Cursor::new("1 .22e0.32 3.21e-3-1.24 1e4"));
        assert_approx_eq!(parser.parse_scalar()?, 1.0);
        assert_approx_eq!(parser.parse_scalar()?, 0.22);
        assert_approx_eq!(parser.parse_scalar()?, 0.32);
        assert_approx_eq!(parser.parse_scalar()?, 3.21e-3);
        assert_approx_eq!(parser.parse_scalar()?, -1.24);
        assert_approx_eq!(parser.parse_scalar()?, 1e4);
        Ok(())
    }

    #[test]
    fn test_parse_transform() -> Result<(), SvgParserError> {
        let tr_str = r#"
            translate(1 2)
            skewX(30deg)
            matrix(1  2 3 4 -3-7)
            scale(2,1)
            rotate(10 1 2)
            rotate(1rad)
        "#;
        let tr_fmt = "matrix(6.56129 5.23393 -1.92617 -2.14614 -5.23999 -4.1231)";
        let tr = Transform::from_str(tr_str)?;
        assert_eq!(format!("{tr:?}"), tr_fmt);
        let tr = Transform::from_str(tr_fmt)?;
        assert_eq!(format!("{tr:?}"), tr_fmt);
        Ok(())
    }
}
