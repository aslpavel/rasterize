//! Utility functions and types used accross the library
use crate::{Scalar, EPSILON, PI};
use std::{fmt, iter::FromIterator, ops::Mul};

/// Restrict value to a certain interval
#[inline]
pub fn clamp<T>(val: T, min: T, max: T) -> T
where
    T: PartialOrd,
{
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

/// Abstraction over slices used by `ArrayIter`
pub trait Array {
    type Item;
    fn new() -> Self;
    fn size(&self) -> usize;
    fn at(&self, index: usize) -> Option<&Self::Item>;
    fn take(&mut self, index: usize) -> Option<Self::Item>;
    fn put(&mut self, index: usize, value: Self::Item) -> Option<Self::Item>;
}

macro_rules! impl_array(
    ($($size:expr),+) => {
        $(
            impl<T: Copy> Array for [Option<T>; $size] {
                type Item = T;
                fn new() -> Self {
                    [None; $size]
                }
                fn size(&self) -> usize { $size }
                fn at(&self, index: usize) -> Option<&Self::Item> {
                    self.get(index).and_then(|item| item.as_ref())
                }
                fn take(&mut self, index: usize) -> Option<Self::Item> {
                    self[index].take()
                }
                fn put(&mut self, index: usize, value: Self::Item) -> Option<Self::Item> {
                    self[index].replace(value)
                }
            }
        )+
    }
);

impl_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

/// Fixed sized iterator
///
/// This type is similar to a smallvec but it never allocates and just panics
/// if you try to fit more data than its size.
#[derive(Clone, Copy)]
pub struct ArrayIter<A> {
    start: usize,
    end: usize,
    array: A,
}

impl<A> fmt::Debug for ArrayIter<A>
where
    A: Array,
    A::Item: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = fmt.debug_list();
        for index in self.start..self.end {
            self.array.at(index).map(|item| list.entry(item));
        }
        list.finish()?;
        Ok(())
    }
}

impl<A: Array> Default for ArrayIter<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Array> ArrayIter<A> {
    pub fn new() -> Self {
        Self {
            start: 0,
            end: 0,
            array: A::new(),
        }
    }

    /// Push new element to the end of the iterator
    pub fn push(&mut self, item: A::Item) {
        self.array.put(self.end, item);
        self.end += 1;
    }

    /// Check if array iterator is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of uncosumed elements
    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<A: Array> Extend<A::Item> for ArrayIter<A> {
    fn extend<T: IntoIterator<Item = A::Item>>(&mut self, iter: T) {
        for item in iter.into_iter() {
            self.push(item);
        }
    }
}

impl<A> FromIterator<A::Item> for ArrayIter<A>
where
    A: Array,
    A::Item: Copy,
{
    fn from_iter<T: IntoIterator<Item = A::Item>>(iter: T) -> Self {
        let mut array = ArrayIter::<A>::new();
        for item in iter.into_iter() {
            array.push(item);
        }
        array
    }
}

impl<A: Array> Iterator for ArrayIter<A> {
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let item = self.array.take(self.start);
            self.start += 1;
            item
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end - self.start;
        (size, Some(size))
    }
}

impl<A: Array> DoubleEndedIterator for ArrayIter<A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            self.end -= 1;
            self.array.take(self.end)
        } else {
            None
        }
    }
}

/// Square 3x3 matrix
#[derive(Debug, Clone, Copy)]
pub(crate) struct M3x3(pub [Scalar; 9]);

impl Mul<M3x3> for M3x3 {
    type Output = M3x3;

    fn mul(self, other: Self) -> Self::Output {
        let M3x3(a) = self;
        let M3x3(b) = other;
        let mut out = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    out[k + 3 * i] += a[j + 3 * i] * b[k + 3 * j];
                }
            }
        }
        M3x3(out)
    }
}

/// Sqaure 4x4 matrix
#[derive(Debug, Clone, Copy)]
pub(crate) struct M4x4(pub [Scalar; 16]);

impl Mul<M4x4> for M4x4 {
    type Output = M4x4;

    fn mul(self, other: Self) -> Self::Output {
        let M4x4(a) = self;
        let M4x4(b) = other;
        let mut out = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    out[k + 4 * i] += a[j + 4 * i] * b[k + 4 * j];
                }
            }
        }
        M4x4(out)
    }
}

/// Solve quadratic equation `a * t ^ 2 + b * t + c = 0` for `t`
pub(crate) fn quadratic_solve(a: Scalar, b: Scalar, c: Scalar) -> impl Iterator<Item = Scalar> {
    let mut result = ArrayIter::<[Option<Scalar>; 2]>::new();
    if a.abs() < EPSILON {
        if b.abs() > EPSILON {
            result.push(-c / b);
        }
        return result;
    }
    let disc = b * b - 4.0 * a * c;
    if disc.abs() < EPSILON {
        result.push(-b / (2.0 * a));
    } else if disc > 0.0 {
        let sq = disc.sqrt();
        // More stable solution than generic formula:
        // https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        if b >= 0.0 {
            let mul = -b - sq;
            result.push(mul / (2.0 * a));
            result.push(2.0 * c / mul);
        } else {
            let mul = -b + sq;
            result.push(2.0 * c / mul);
            result.push(mul / (2.0 * a));
        }
    }
    result
}

/// Solve cubic equation `a * t ^ 3 + b * t ^ 2 + c * t + d = 0` for `t`
/// Reference: https://www.trans4mind.com/personal_development/mathematics/polynomials/cubicAlgebra.htm
#[allow(clippy::many_single_char_names)]
pub(crate) fn cubic_solve(
    a: Scalar,
    b: Scalar,
    c: Scalar,
    d: Scalar,
) -> impl Iterator<Item = Scalar> {
    let mut results = ArrayIter::<[Option<Scalar>; 3]>::new();
    if a.abs() < 1.0 && a.abs().powi(2) < EPSILON {
        results.extend(quadratic_solve(b, c, d));
        return results;
    }
    if d.abs() < EPSILON {
        results.push(0.0);
        results.extend(quadratic_solve(a, b, c));
        return results;
    }

    // helper to calculate cubic root
    fn crt(value: Scalar) -> Scalar {
        if value < 0.0 {
            -(-value).powf(1.0 / 3.0)
        } else {
            value.powf(1.0 / 3.0)
        }
    }

    // convert to `t ^ 3 + a * t ^ 2 + b * t + c = 0`
    let (a, b, c) = (b / a, c / a, d / a);

    // convert to `t ^ 3 + p * t + q = 0`
    let p = (3.0 * b - a * a) / 3.0;
    let q = ((2.0 * a * a - 9.0 * b) * a + 27.0 * c) / 27.0;
    let p3 = p / 3.0;
    let q2 = q / 2.0;
    let disc = q2 * q2 + p3 * p3 * p3;

    if disc.abs() < EPSILON {
        // two roots
        let u1 = if q2 < 0.0 { crt(-q2) } else { -crt(q2) };
        results.push(2.0 * u1 - a / 3.0);
        results.push(-u1 - a / 3.0);
    } else if disc > 0.0 {
        // one root
        let sd = disc.sqrt();
        results.push(crt(sd - q2) - crt(sd + q2) - a / 3.0);
    } else {
        // three roots
        let r = (-p3 * p3 * p3).sqrt();
        let phi = clamp(-q / (2.0 * r), -1.0, 1.0).acos();
        let c = 2.0 * crt(r);
        let a3 = a / 3.0;
        results.push(c * (phi / 3.0).cos() - a3);
        results.push(c * ((phi + 2.0 * PI) / 3.0).cos() - a3);
        results.push(c * ((phi + 4.0 * PI) / 3.0).cos() - a3);
    }

    results
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[macro_export]
    macro_rules! assert_approx_eq {
        ( $v0:expr, $v1: expr ) => {{
            assert!(($v0 - $v1).abs() < $crate::EPSILON, "{} != {}", $v0, $v1);
        }};
        ( $v0:expr, $v1: expr, $e: expr ) => {{
            assert!(($v0 - $v1).abs() < $e, "{} != {}", $v0, $v1);
        }};
    }

    #[test]
    fn test_solve() {
        fn solve_check(a: Scalar, b: Scalar, c: Scalar, d: Scalar, roots: &[Scalar]) {
            const PREC: Scalar = 0.00001;
            let mut index = 0;
            for root in cubic_solve(a, b, c, d) {
                let value = a * root * root * root + b * root * root + c * root + d;
                if value.abs() > PREC {
                    panic!("f(x = {}) = {} != 0", root, value);
                }
                match roots.get(index) {
                    Some(root_ref) => assert_approx_eq!(root, *root_ref, PREC),
                    None => panic!("result is longer than expected: {:?}", roots),
                }
                index += 1;
            }
            if index != roots.len() {
                panic!("result is shorter than expected: {:?}", roots)
            }
        }

        // cubic
        solve_check(1.0, 0.0, -12.0, 16.0, &[-4.0, 2.0]);
        solve_check(1.0, -6.0, 11.0, -6.0, &[3.0, 1.0, 2.0]);
        solve_check(23.0, 17.0, -11.0, 13.0, &[-1.38148]);

        // quadratic
        solve_check(0.0, 1.0, -5.0, 6.0, &[2.0, 3.0]);
        solve_check(0.0, 1.0, -6.0, 9.0, &[3.0]);
        solve_check(0.0, 1.0, 3.0, 5.0, &[]);

        // liner
        solve_check(0.0, 0.0, 5.0, 10.0, &[-2.0]);
    }

    #[test]
    fn test_matmul() {
        let m0 = M3x3([31.0, 11.0, 21.0, 12.0, 19.0, 3.0, 18.0, 25.0, 16.0]);
        let m1 = M3x3([19.0, 7.0, 14.0, 1.0, 0.0, 12.0, 10.0, 29.0, 29.0]);
        let r = [
            810.0, 826.0, 1175.0, 277.0, 171.0, 483.0, 527.0, 590.0, 1016.0,
        ];
        for (v0, v1) in (m0 * m1).0.iter().zip(&r) {
            assert_approx_eq!(v0, v1);
        }
    }

    #[test]
    fn test_array_iter() {
        let mut iter: ArrayIter<[Option<u32>; 5]> = (0..5).collect();
        assert_eq!(iter.len(), 5);
        assert!(!iter.is_empty());
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next_back(), Some(4));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next_back(), Some(3));
        assert_eq!(iter.next_back(), Some(2));
        assert_eq!(iter.len(), 0);
        assert!(iter.is_empty());
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }
}
