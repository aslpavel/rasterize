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

    // put byte into input buffer, at most one byte is cached
    fn unparse_byte(&mut self, byte: u8) {
        debug_assert!(self.input_buffer.is_none());
        self.input_buffer = Some(byte);
    }

    // consume input while `pred` predicate is true, consumed input is stored in `Self::buffer`
    fn parse_while(
        &mut self,
        mut pred: impl FnMut(u8) -> bool,
        mut proc: impl FnMut(u8),
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
            proc(byte);
        }
        Ok(count)
    }

    // consume at most one byte from the input, if predicate returns true
    fn parse_once(
        &mut self,
        pred: impl FnOnce(u8) -> bool,
        proc: impl FnOnce(u8),
    ) -> Result<bool, SVGPathParserError> {
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
            |byte| matches!(byte, b'0'..=b'9'),
            |byte| push_digit(&mut mantissa, byte),
        )?;
        let frac = if self.parse_once(|byte| matches!(byte, b'.'), |_| {})? {
            self.parse_while(
                |byte| matches!(byte, b'0'..=b'9'),
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
            return Err(SVGPathParserError::InvalidScalar);
        }

        if self.parse_once(|byte| matches!(byte, b'e' | b'E'), |_| {})? {
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
                |byte| matches!(byte, b'0'..=b'9'),
                |byte| push_digit(&mut sci, byte),
            )? == 0
            {
                return Err(SVGPathParserError::InvalidScalar);
            }
            exponent = exponent.wrapping_add(sci_sign * sci)
        }

        Ok((mantissa as Scalar) * (10.0 as Scalar).powi(exponent as i32))
    }

    // parse pair of scalars and convert it to a point
    fn parse_point(&mut self) -> Result<Point, SVGPathParserError> {
        let point = Point::new(self.parse_scalar()?, self.parse_scalar()?);
        match self.prev_op {
            Some(cmd) if cmd.is_ascii_lowercase() => Ok(point + self.position),
            _ => Ok(point),
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

/// Error while parsing path in the SVG format
#[derive(Debug)]
pub enum SVGPathParserError {
    /// Failed to parse SVG command
    InvalidCmd(u8),
    /// Failed to parse scalar value
    InvalidScalar,
    /// Failed to parse flag value
    InvalidFlag,
    /// Unexpected segment type found while parsing curve segment
    UnexpectedSegmentType,
    /// IO error propagated while reading input stream
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;
    use std::io::Cursor;

    #[test]
    fn test_parse_scalar() -> Result<(), SVGPathParserError> {
        let mut parser = SVGPathParser::new(Cursor::new("1 .22e0.32 3.21e-3-1.24 1e4"));
        assert_approx_eq!(parser.parse_scalar()?, 1.0);
        assert_approx_eq!(parser.parse_scalar()?, 0.22);
        assert_approx_eq!(parser.parse_scalar()?, 0.32);
        assert_approx_eq!(parser.parse_scalar()?, 3.21e-3);
        assert_approx_eq!(parser.parse_scalar()?, -1.24);
        assert_approx_eq!(parser.parse_scalar()?, 1e4);
        Ok(())
    }
}
