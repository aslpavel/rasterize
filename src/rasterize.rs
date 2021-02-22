//! Rasterization logic
//!
//! This module include two rasterizers:
//!   - Signed difference rasterizer
//!   - Active-Edge rasterizer
//!
//! ## Signed difference rasterizer
//! It works in two steps, first going over all lines and writing signed difference
//! (difference value represents how to adjust winding number from one pixel to another).
//! And on the second step it goes over all pixels and accumulates winding number, and
//! depending on the fill rule produces alpha map of the path.
//!
//! Features:
//!  - this method is fast
//!  - requires memory equal to the size of the image
//!
//! ## Active-Edge-Table rasterizer
//! This method is based on the data stracture Edge-Table which keeps all lines ordered by
//! lower y coordinate, and then scanning over all pixels line by line, once lower pixel of a
//! line is reached line is activated and put into Active-Edge-Table, and later deactivated once
//! once scan line convers point with the highest y coordinate.
//!
//! Reference: Computer graphics principles and practice (by Foley) 3.6 Filling Polygons.
//!
//! Features:
//!  - this method is slower
//!  - but requires less memory
use crate::{
    Curve, FillRule, Line, Path, Point, Scalar, SurfaceMut, Transform, DEFAULT_FLATNESS, EPSILON,
};
use std::{cmp::min, collections::VecDeque};

pub trait Rasterizer {
    fn rasterize(
        &self,
        path: &Path,
        tr: Transform,
        surf: &mut dyn SurfaceMut<Item = Scalar>,
        fill_rule: FillRule,
    );

    fn name(&self) -> &str;
}

impl<'a, R: Rasterizer> Rasterizer for &'a R {
    fn rasterize(
        &self,
        path: &Path,
        tr: Transform,
        surf: &mut dyn SurfaceMut<Item = Scalar>,
        fill_rule: FillRule,
    ) {
        (**self).rasterize(path, tr, surf, fill_rule)
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}

impl Rasterizer for Box<dyn Rasterizer> {
    fn rasterize(
        &self,
        path: &Path,
        tr: Transform,
        surf: &mut dyn SurfaceMut<Item = Scalar>,
        fill_rule: FillRule,
    ) {
        (**self).rasterize(path, tr, surf, fill_rule)
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}

pub struct SignedDifferenceRasterizer {
    flatness: Scalar,
}

impl Default for SignedDifferenceRasterizer {
    fn default() -> Self {
        Self {
            flatness: DEFAULT_FLATNESS,
        }
    }
}

impl Rasterizer for SignedDifferenceRasterizer {
    fn rasterize(
        &self,
        path: &Path,
        tr: Transform,
        surf: &mut dyn SurfaceMut<Item = Scalar>,
        fill_rule: FillRule,
    ) {
        for line in path.flatten(tr, self.flatness, true) {
            signed_difference_line(&mut *surf, line);
        }
        signed_difference_to_mask(surf, fill_rule);
    }

    fn name(&self) -> &str {
        "signed-difference"
    }
}

/// Update provided surface with the signed difference of the line
///
/// Signed difference is a diffrence between adjacent pixels introduced by the line.
fn signed_difference_line(mut surf: impl SurfaceMut<Item = Scalar>, line: Line) {
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

/// Conver signed difference surface to a mask
fn signed_difference_to_mask(mut surf: impl SurfaceMut<Item = Scalar>, fill_rule: FillRule) {
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

/// Active-Edge rasterizer
///
/// This method is based on the data stracture Edge-Table which keeps all lines ordered by
/// lower y coordinate, and then scanning over all pixels line by line, once lower pixel of a
/// line is reached line is activated and put into Active-Edge-Table, and later deactivated once
/// once scan line convers point with the highest y coordinate.
///
/// Reference: Computer graphics principles and practice (by Foley) 3.6 Filling Polygons.
pub struct ActiveEdgeRasterizer {
    flatness: Scalar,
}

impl Default for ActiveEdgeRasterizer {
    fn default() -> Self {
        Self {
            flatness: DEFAULT_FLATNESS,
        }
    }
}

impl Rasterizer for ActiveEdgeRasterizer {
    fn rasterize(
        &self,
        path: &Path,
        tr: Transform,
        surf: &mut dyn SurfaceMut<Item = Scalar>,
        fill_rule: FillRule,
    ) {
        let shape = surf.shape();
        let data = surf.data_mut();
        for pixel in ActiveEdgeIter::new(
            shape.width,
            shape.height,
            fill_rule,
            path.flatten(tr, self.flatness, true),
        ) {
            data[shape.offset(pixel.y, pixel.x)] = pixel.alpha;
        }
    }

    fn name(&self) -> &str {
        "active-edge"
    }
}

pub struct ActiveEdgeIter {
    // all edges sorted by `Edge::row` in descending order
    edge_inactive: Vec<Vec<Edge>>,
    // once scanline touches and it is activated and put into this list
    edge_active: VecDeque<Edge>,
    // row iterators are created for all active edges on
    iters_inactive: Vec<EdgeRowIter>,
    // accumulated iterator over all active row iterators
    iters_active: EdgeAccIter,
    // currently accumulated winding number
    winding: Scalar,
    // fill rule to be used
    fill_rule: FillRule,
    // width of the output image
    width: usize,
    // height of the output image
    height: usize,
    // current column (x - coordindate)
    column: usize,
    // current row (y - coordinate)
    row: usize,
}

impl ActiveEdgeIter {
    pub fn new(
        width: usize,
        height: usize,
        fill_rule: FillRule,
        lines: impl Iterator<Item = Line>,
    ) -> Self {
        let mut edge_table: Vec<Vec<Edge>> = Vec::new();
        edge_table.resize_with(height, Default::default);
        for line in lines {
            if let Some(edge) = Edge::new(line) {
                if edge.row >= height {
                    continue;
                }
                edge_table[height - edge.row - 1].push(edge);
            }
        }
        let mut this = Self {
            edge_inactive: edge_table,
            edge_active: Default::default(),
            iters_inactive: Default::default(),
            iters_active: EdgeAccIter::new(),
            winding: 0.0,
            fill_rule,
            width,
            height,
            column: 0,
            row: 0,
        };
        this.next_row();
        this
    }

    /// Swith to the next row
    fn next_row(&mut self) {
        // clear all iterators
        self.iters_inactive.clear();
        self.iters_active.clear();

        // create new row iterators for all active edges
        for _ in 0..self.edge_active.len() {
            if let Some(edge) = self.edge_active.pop_front() {
                if let Some((edge, iter)) = edge.next_row() {
                    self.iters_inactive.push(iter);
                    self.edge_active.push_back(edge);
                }
            }
        }

        // activate new edges
        if let Some(edges) = self.edge_inactive.pop() {
            for edge in edges {
                if let Some((edge, iter)) = edge.next_row() {
                    self.iters_inactive.push(iter);
                    self.edge_active.push_back(edge);
                }
            }
        }

        // sort iterator by column
        self.iters_inactive
            .sort_unstable_by(|a, b| b.column.cmp(&a.column));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Pixel {
    pub x: usize,
    pub y: usize,
    pub alpha: Scalar,
}

impl Iterator for ActiveEdgeIter {
    type Item = Pixel;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row > self.height {
                return None;
            }

            // find activated iterators
            while let Some(iter) = self.iters_inactive.pop() {
                if iter.column <= self.column {
                    self.iters_active.push(iter);
                } else {
                    self.iters_inactive.push(iter);
                    break;
                }
            }

            // progress active row iterators
            match self.iters_active.next() {
                Some(winding_delta) => self.winding += winding_delta,
                None if self.winding.abs() < 1e-6 => {
                    // skip forward to the first activated iterator
                    match self.iters_inactive.last() {
                        Some(iter) if iter.column < self.width => {
                            self.column = iter.column;
                        }
                        _ => {
                            self.column = 0;
                            self.row += 1;
                            self.winding = 0.0;
                            self.next_row();
                        }
                    }
                    continue;
                }
                None => {}
            }

            // self.winding += self.iters_active.next().unwrap_or(0.0);
            let pixel = Pixel {
                x: self.column,
                y: self.row,
                alpha: self.fill_rule.alpha_from_winding(self.winding),
            };

            // update postion
            self.column += 1;
            if self.column >= self.width {
                self.column = 0;
                self.row += 1;
                self.winding = 0.0;
                self.next_row();
            }

            return Some(pixel);
        }
    }
}

#[inline]
fn next_ceil(value: Scalar) -> Scalar {
    value.floor() + 1.0
}

/// Edge represents unconsumed part of the line segments
///
/// `Edge::line` is always directed from point with lower to higher `y` coordinate.
#[derive(Debug)]
struct Edge {
    // unconsumed part of the line segment
    line: Line,
    // `dx/dy` slope of the edge
    dxdy: Scalar,
    // `dy/dx` slope of the edge, it empty for vertial lines
    dydx: Option<Scalar>,
    // `1.0` if edge is going from lower to higher `y`, `-1.0` otherwise
    dir: Scalar,
    // first row that will be effected by this edge
    row: usize,
}

impl Edge {
    /// Create edge from the line
    ///
    /// Returns constructed `Edge` and lowest row effected by it.
    fn new(line: Line) -> Option<Self> {
        let Line([p0, p1]) = line;
        if (p0.y() - p1.y()).abs() < EPSILON {
            // horizontal lines have no effect
            return None;
        }
        // order from lower to higher y coordinate
        let (dir, p0, p1) = if p0.y() <= p1.y() {
            (1.0, p0, p1)
        } else {
            (-1.0, p1, p0)
        };
        let dxdy = (p1.x() - p0.x()) / (p1.y() - p0.y());
        let dydx = (dxdy.abs() > EPSILON).then(|| dxdy.recip());
        // throw away part with negative `y`
        let p0 = if p0.y() < 0.0 {
            Point::new(p0.x() - p0.y() * dxdy, 0.0)
        } else {
            p0
        };
        Some(Self {
            line: Line::new(p0, p1),
            dxdy,
            dydx,
            dir,
            row: p0.y().floor() as usize,
        })
    }

    /// Split edge into row iterator and reminder of the edge not convered by
    /// the row iterator.
    fn next_row(self) -> Option<(Edge, EdgeRowIter)> {
        EdgeRowIter::new(self)
    }
}

/// Iterator that calculates winding difference introduced by the line
///
/// This iterator is activated once rasterizer reaches `EdgeRowIter::column`,
/// and for each subsequent pixel `EdgeRowIter::next` is called, which returns
/// winding difference introduce by the line.
#[derive(Debug)]
struct EdgeRowIter {
    // line segment that will effect winding number of current
    line: Option<Line>,
    // difference introduced by previous pixel
    reminder: Option<Scalar>,
    // `dy/dx` slope of the line
    dydx: Option<Scalar>,
    // first effected column
    column: usize,
    // direction of the line
    dir: Scalar,
}

impl EdgeRowIter {
    fn new(mut edge: Edge) -> Option<(Edge, Self)> {
        let Line([p0, p1]) = edge.line;
        if p1.y() - p0.y() < EPSILON {
            // edge is fully consumed and should be removed
            return None;
        }

        // intersection with lower edge of the next row
        let y_split = next_ceil(p0.y()).min(p1.y());
        let x_split = p0.x() + edge.dxdy * (y_split - p0.y());
        let p_split = Point::new(x_split, y_split);

        // reduce the size of the edge
        edge.line = Line::new(p_split, p1);

        let (dir, line) = if p0.x() <= p_split.x() {
            (edge.dir, Line::new(p0, p_split))
        } else {
            (-edge.dir, Line::new(p_split, p0))
        };
        let iter = Self {
            line: Some(line),
            reminder: None,
            dydx: edge.dydx,
            column: line.start().x() as usize,
            dir,
        };
        Some((edge, iter))
    }
}

impl Iterator for EdgeRowIter {
    type Item = Scalar;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(Line([p0, p1])) = self.line.take() {
            let x_ceil = next_ceil(p0.x());
            let x = x_ceil.min(p1.x());
            let h = match self.dydx {
                Some(dydx) => dydx * (x - p0.x()),
                None => next_ceil(p0.y()).min(p1.y()) - p0.y(),
            };
            let y = p0.y() + h;
            let s0 = (2.0 * x_ceil - x - p0.x()) * h / 2.0;
            let s1 = h - s0;
            if p1.x() > x {
                self.line = Some(Line::new((x, y), p1));
            }
            let s_prev = self.reminder.replace(self.dir * s1).unwrap_or(0.0);
            Some(self.dir * s0 + s_prev)
        } else {
            self.reminder.take()
        }
    }
}

/// Accumulator iterator
///
/// Sums all values returned by sub-iterators and returns it as an item.
struct EdgeAccIter {
    iters: VecDeque<EdgeRowIter>,
}

impl EdgeAccIter {
    fn new() -> Self {
        Self {
            iters: Default::default(),
        }
    }

    // push new row iterator
    fn push(&mut self, iter: EdgeRowIter) {
        self.iters.push_back(iter)
    }

    // remove all row iterator
    fn clear(&mut self) {
        self.iters.clear()
    }
}

impl Iterator for EdgeAccIter {
    type Item = Scalar;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iters.is_empty() {
            return None;
        }
        let mut acc = 0.0;
        for _ in 0..self.iters.len() {
            if let Some(mut iter) = self.iters.pop_front() {
                if let Some(value) = iter.next() {
                    acc += value;
                    self.iters.push_back(iter);
                }
            }
        }
        Some(acc)
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

    #[test]
    fn test_edge_iter() {
        let line = Line::new((0.0, 0.0), (6.0, 2.0));
        let edge = Edge::new(line).unwrap();
        assert_eq!(edge.row, 0);
        // first row
        let (edge, mut iter) = edge.next_row().unwrap();
        assert_eq!(iter.column, 0);
        assert_approx_eq!(iter.next().unwrap(), 1.0 / 6.0);
        assert_approx_eq!(iter.next().unwrap(), 2.0 / 6.0);
        assert_approx_eq!(iter.next().unwrap(), 2.0 / 6.0);
        assert_approx_eq!(iter.next().unwrap(), 1.0 / 6.0);
        assert!(iter.next().is_none());
        // second row
        let (edge, iter) = edge.next_row().unwrap();
        assert_eq!(iter.column, 3);
        assert_approx_eq!(iter.sum::<Scalar>(), 1.0);
        // should not return next row
        assert!(edge.next_row().is_none());

        let line = Line::new((1.0, 1.0), (4.0, 0.0));
        let edge = Edge::new(line).unwrap();
        assert_eq!(edge.row, 0);
        // first row
        let (edge, mut iter) = edge.next_row().unwrap();
        assert_eq!(iter.column, 1);
        assert_approx_eq!(iter.next().unwrap(), -1.0 / 6.0);
        assert_approx_eq!(iter.next().unwrap(), -2.0 / 6.0);
        assert_approx_eq!(iter.next().unwrap(), -2.0 / 6.0);
        assert_approx_eq!(iter.next().unwrap(), -1.0 / 6.0);
        // shoud not return next row
        assert!(edge.next_row().is_none());

        // vertical line
        let line = Line::new((0.5, 0.5), (0.5, 1.5));
        let edge = Edge::new(line).unwrap();
        assert_eq!(edge.row, 0);
        // first row
        let (edge, mut iter) = edge.next_row().unwrap();
        assert_eq!(iter.column, 0);
        assert_approx_eq!(iter.next().unwrap(), 1.0 / 4.0);
        assert_approx_eq!(iter.next().unwrap(), 1.0 / 4.0);
        assert!(iter.next().is_none());
        // second row
        let (edge, iter) = edge.next_row().unwrap();
        assert_approx_eq!(iter.sum::<Scalar>(), 0.5);
        // should not return next row
        assert!(edge.next_row().is_none());
    }
}
