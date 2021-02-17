use crate::{FillRule, Line, Point, Scalar, SurfaceMut, EPSILON};
use std::cmp::min;

/// Update provided surface with the signed difference of the line
///
/// Signed difference is a diffrence between adjacent pixels introduced by the line.
pub(crate) fn signed_difference_line(mut surf: impl SurfaceMut<Item = Scalar>, line: Line) {
    // y - is a row
    // x - is a column
    let Line([p0, p1]) = line;

    // handle lines that are intersecting `x == surf.width()`
    // - just throw away part that has x > surf.width for all points
    let width = surf.width() as Scalar - 1.0;
    let line = if p0.x() > width || p1.x() > width {
        if p0.x() > width && p1.x() > width {
            Line::new((width - 0.001, p0.y()), (width - 0.001, p1.y()))
        } else {
            let t = (p0.x() - width) / (p0.x() - p1.x());
            let mid = Point::new(width, (1.0 - t) * p0.y() + t * p1.y());
            if p0.x() < width {
                Line::new(p0, mid)
            } else {
                Line::new(mid, p1)
            }
        }
    } else {
        line
    };

    // handle lines that are intersecting `x == 0.0`
    // - line is splitted in left (for all points where x < 0.0) and the mid part
    // - left part is converted to a vertical line that spans same y's and x == 0.0
    // - left part is rastterized recursivelly, and mid part rasterized after this
    let line = if p0.x() < 0.0 || p1.x() < 0.0 {
        let (vertical, line) = if p1.x() > 0.0 || p0.x() > 0.0 {
            let t = p0.x() / (p0.x() - p1.x());
            let mid = Point::new(0.0, (1.0 - t) * p0.y() + t * p1.y());
            if p1.x() > 0.0 {
                let p = Point::new(0.0, p0.y());
                (Line::new(p, mid), Line::new(mid, p1))
            } else {
                let p = Point::new(0.0, p1.y());
                (Line::new(mid, p), Line::new(p0, mid))
            }
        } else {
            (
                Line::new((0.0, p0.y()), (0.0, p1.y())),
                Line::new((0.0, 0.0), (0.0, 0.0)),
            )
        };
        // signed difference by the line left of `x == 0.0`
        signed_difference_line(surf.view_mut(.., ..), vertical);
        line
    } else {
        line
    };

    let Line([p0, p1]) = line;
    let shape = surf.shape();
    let data = surf.data_mut();
    let stride = shape.col_stride;

    if (p0.y() - p1.y()).abs() < EPSILON {
        // line does not introduce any signed converage
        return;
    }
    // always iterate from the point with the smallest y coordinate
    let (dir, p0, p1) = if p0.y() < p1.y() {
        (1.0, p0, p1)
    } else {
        (-1.0, p1, p0)
    };
    let dxdy = (p1.x() - p0.x()) / (p1.y() - p0.y());
    // find first point to trace. since we are going to interate over y's
    // we should pick min(y , p0.y) as a starting y point, and adjust x
    // accordingly
    let y = p0.y().max(0.0) as usize;
    let mut x = if p0.y() < 0.0 {
        p0.x() - p0.y() * dxdy
    } else {
        p0.x()
    };
    let mut x_next = x;
    for y in y..min(shape.height, p1.y().ceil().max(0.0) as usize) {
        x = x_next;
        let row_offset = shape.offset(y, 0); // current line offset in the data array
        let dy = ((y + 1) as Scalar).min(p1.y()) - (y as Scalar).max(p0.y());
        // signed y difference
        let d = dir * dy;
        // find next x position
        x_next = x + dxdy * dy;
        // order (x, x_next) from smaller value x0 to bigger x1
        let (x0, x1) = if x < x_next { (x, x_next) } else { (x_next, x) };
        // lower bound of effected x pixels
        let x0_floor = x0.floor().max(0.0);
        let x0i = x0_floor as i32;
        // uppwer bound of effected x pixels
        let x1_ceil = x1.ceil();
        let x1i = x1_ceil as i32;
        if x1i <= x0i + 1 {
            // only goes through one pixel (with the total coverage of `d` spread over two pixels)
            let xmf = 0.5 * (x + x_next) - x0_floor; // effective height
            data[row_offset + (x0i as usize) * stride] += d * (1.0 - xmf);
            data[row_offset + ((x0i + 1) as usize) * stride] += d * xmf;
        } else {
            let s = (x1 - x0).recip();
            let x0f = x0 - x0_floor; // fractional part of x0
            let x1f = x1 - x1_ceil + 1.0; // fractional part of x1
            let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f); // fractional area of the pixel with smallest x
            let am = 0.5 * s * x1f * x1f; // fractional area of the pixel with largest x
            data[row_offset + (x0i as usize) * stride] += d * a0;
            if x1i == x0i + 2 {
                // only two pixels are covered
                data[row_offset + ((x0i + 1) as usize) * stride] += d * (1.0 - a0 - am);
            } else {
                // second pixel
                let a1 = s * (1.5 - x0f);
                data[row_offset + ((x0i + 1) as usize) * stride] += d * (a1 - a0);
                // (second, last) pixels
                for xi in x0i + 2..x1i - 1 {
                    data[row_offset + (xi as usize) * stride] += d * s;
                }
                // last pixel
                let a2 = a1 + (x1i - x0i - 3) as Scalar * s;
                data[row_offset + ((x1i - 1) as usize) * stride] += d * (1.0 - a2 - am);
            }
            data[row_offset + (x1i as usize) * stride] += d * am
        }
    }
}

pub(crate) fn signed_difference_to_mask(
    mut surf: impl SurfaceMut<Item = Scalar>,
    fill_rule: FillRule,
) {
    let shape = surf.shape();
    let data = surf.data_mut();
    match fill_rule {
        FillRule::NonZero => {
            for y in 0..shape.height {
                let mut acc = 0.0;
                for x in 0..shape.width {
                    let offset = shape.offset(y, x);
                    acc += data[offset];

                    let value = acc.abs();
                    data[offset] = if value > 1.0 {
                        1.0
                    } else if value < 1e-6 {
                        0.0
                    } else {
                        value
                    };
                }
            }
        }
        FillRule::EvenOdd => {
            for y in 0..shape.height {
                let mut acc = 0.0;
                for x in 0..shape.width {
                    let offset = shape.offset(y, x);
                    acc += data[offset];

                    data[offset] = ((acc + 1.0).rem_euclid(2.0) - 1.0).abs()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq, Surface, SurfaceOwned};

    #[test]
    fn test_signed_difference_line() {
        let mut surf = SurfaceOwned::new(2, 5);

        // line convers many columns but just one row
        signed_difference_line(&mut surf, Line::new((0.5, 1.0), (3.5, 0.0)));
        // covered areas per-pixel
        let a0 = (0.5 * (1.0 / 6.0)) / 2.0;
        let a1 = ((1.0 / 6.0) + (3.0 / 6.0)) / 2.0;
        let a2 = ((3.0 / 6.0) + (5.0 / 6.0)) / 2.0;
        assert_approx_eq!(*surf.get(0, 0).unwrap(), -a0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), a0 - a1);
        assert_approx_eq!(*surf.get(0, 2).unwrap(), a1 - a2);
        assert_approx_eq!(*surf.get(0, 3).unwrap(), a0 - a1);
        assert_approx_eq!(*surf.get(0, 4).unwrap(), -a0);
        // total difference
        let a: Scalar = surf.iter().sum();
        assert_approx_eq!(a, -1.0);
        surf.clear();

        // out of bound line (intersects x = 0.0)
        signed_difference_line(&mut surf, Line::new((-1.0, 0.0), (1.0, 1.0)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 3.0 / 4.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 / 4.0);
        surf.clear();

        // multiple rows diag
        signed_difference_line(&mut surf, Line::new((0.0, -0.5), (2.0, 1.5)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 - 2.0 / 8.0);
        assert_approx_eq!(*surf.get(0, 2).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 1).unwrap(), 1.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 2).unwrap(), 0.5 - 1.0 / 8.0);
        surf.clear();

        // only two pixels covered
        signed_difference_line(&mut surf, Line::new((0.1, 0.1), (1.9, 0.9)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 0.18);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 0.44);
        assert_approx_eq!(*surf.get(0, 2).unwrap(), 0.18);
        surf.clear();

        // single pixel covered
        signed_difference_line(&mut surf, Line::new((0.1, 0.1), (0.9, 0.9)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 0.4);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 0.8 - 0.4);
        surf.clear();

        // multiple rows vertical
        signed_difference_line(&mut surf, Line::new((0.5, 0.5), (0.5, 1.75)));
        assert_approx_eq!(*surf.get(0, 0).unwrap(), 1.0 / 4.0);
        assert_approx_eq!(*surf.get(0, 1).unwrap(), 1.0 / 4.0);
        assert_approx_eq!(*surf.get(1, 0).unwrap(), 3.0 / 8.0);
        assert_approx_eq!(*surf.get(1, 1).unwrap(), 3.0 / 8.0);
        surf.clear();
    }
}
