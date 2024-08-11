//! Utility functions and types used across the library
use crate::{Scalar, EPSILON, PI};
use std::{fmt, iter::FusedIterator, ops::Mul};

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

#[cold]
#[inline(never)]
fn cold() {}

/// Hint to the compiler that boolean likely false
#[inline(always)]
pub fn unlikely(b: bool) -> bool {
    if b {
        cold()
    }
    b
}

/// Hint to the compiler that boolean likely true
#[inline(always)]
pub fn likely(b: bool) -> bool {
    if !b {
        cold()
    }
    b
}

/// Fixed sized iterator
///
/// This type is similar to a `smallvec` but it never allocates and just panics
/// if you try to fit more data than its size.
#[derive(Clone, Copy)]
pub struct ArrayIter<T, const SIZE: usize> {
    start: usize,
    end: usize,
    items: [Option<T>; SIZE],
}

impl<T, const SIZE: usize> ArrayIter<T, SIZE> {
    pub fn new() -> Self
    where
        T: Copy,
    {
        Self {
            start: 0,
            end: 0,
            // remove Copy constraint everywhere when Default::default() is implemented
            // for arrays with size bigger than 32
            items: [None; SIZE],
        }
    }

    /// Push new element to the end of the iterator
    ///
    /// Panics if size is exceeded
    pub fn push(&mut self, item: T) {
        self.items[self.end] = Some(item);
        self.end += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn into_array(self) -> [Option<T>; SIZE] {
        self.items
    }
}

impl<T: Copy, const SIZE: usize> Default for ArrayIter<T, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const SIZE: usize> fmt::Debug for ArrayIter<T, SIZE>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = fmt.debug_list();
        for index in self.start..self.end {
            if let Some(item) = &self.items[index] {
                list.entry(item);
            }
        }
        list.finish()?;
        Ok(())
    }
}

impl<T, const SIZE: usize> Extend<T> for ArrayIter<T, SIZE> {
    fn extend<TS: IntoIterator<Item = T>>(&mut self, iter: TS) {
        for item in iter.into_iter() {
            self.push(item);
        }
    }
}

impl<T: Copy, const SIZE: usize> FromIterator<T> for ArrayIter<T, SIZE> {
    fn from_iter<TS: IntoIterator<Item = T>>(iter: TS) -> Self {
        let mut array: Self = ArrayIter::new();
        array.extend(iter);
        array
    }
}

impl<T, const SIZE: usize> Iterator for ArrayIter<T, SIZE> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let item = self.items[self.start].take();
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

impl<T, const SIZE: usize> DoubleEndedIterator for ArrayIter<T, SIZE> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            self.end -= 1;
            self.items[self.end].take()
        } else {
            None
        }
    }
}

impl<T, const SIZE: usize> ExactSizeIterator for ArrayIter<T, SIZE> {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<T, const SIZE: usize> FusedIterator for ArrayIter<T, SIZE> {}

/// Square 3x3 matrix
#[derive(Debug, Clone, Copy)]
pub struct M3x3(pub [Scalar; 9]);

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

/// Square 4x4 matrix
#[derive(Debug, Clone, Copy)]
pub struct M4x4(pub [Scalar; 16]);

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
pub fn quadratic_solve(a: Scalar, b: Scalar, c: Scalar) -> ArrayIter<Scalar, 2> {
    let mut result = ArrayIter::<Scalar, 2>::new();
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
/// Reference: <https://www.trans4mind.com/personal_development/mathematics/polynomials/cubicAlgebra.htm>
#[allow(clippy::many_single_char_names)]
pub fn cubic_solve(a: Scalar, b: Scalar, c: Scalar, d: Scalar) -> ArrayIter<Scalar, 3> {
    let mut results = ArrayIter::<Scalar, 3>::new();
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

// 16 point quadrature (weight, position)
// [Legendre Gauss Qaudrature](https://pomax.github.io/bezierinfo/legendre-gauss.html)
pub const QUADRATURE_32: [(Scalar, Scalar); 32] = [
    (0.0965400885147278, -0.0483076656877383),
    (0.0965400885147278, 0.0483076656877383),
    (0.0956387200792749, -0.1444719615827965),
    (0.0956387200792749, 0.1444719615827965),
    (0.0938443990808046, -0.2392873622521371),
    (0.0938443990808046, 0.2392873622521371),
    (0.0911738786957639, -0.3318686022821277),
    (0.0911738786957639, 0.3318686022821277),
    (0.0876520930044038, -0.4213512761306353),
    (0.0876520930044038, 0.4213512761306353),
    (0.0833119242269467, -0.5068999089322294),
    (0.0833119242269467, 0.5068999089322294),
    (0.0781938957870703, -0.5877157572407623),
    (0.0781938957870703, 0.5877157572407623),
    (0.0723457941088485, -0.6630442669302152),
    (0.0723457941088485, 0.6630442669302152),
    (0.0658222227763618, -0.7321821187402897),
    (0.0658222227763618, 0.7321821187402897),
    (0.0586840934785355, -0.7944837959679424),
    (0.0586840934785355, 0.7944837959679424),
    (0.0509980592623762, -0.84936761373257),
    (0.0509980592623762, 0.84936761373257),
    (0.0428358980222267, -0.8963211557660521),
    (0.0428358980222267, 0.8963211557660521),
    (0.0342738629130214, -0.9349060759377397),
    (0.0342738629130214, 0.9349060759377397),
    (0.0253920653092621, -0.9647622555875064),
    (0.0253920653092621, 0.9647622555875064),
    (0.0162743947309057, -0.9856115115452684),
    (0.0162743947309057, 0.9856115115452684),
    (0.0070186100094701, -0.9972638618494816),
    (0.0070186100094701, 0.9972638618494816),
];

pub const QUADRATURE_16: [(Scalar, Scalar); 16] = [
    (0.1894506104550685, -0.0950125098376374),
    (0.1894506104550685, 0.0950125098376374),
    (0.1826034150449236, -0.2816035507792589),
    (0.1826034150449236, 0.2816035507792589),
    (0.1691565193950025, -0.4580167776572274),
    (0.1691565193950025, 0.4580167776572274),
    (0.1495959888165767, -0.6178762444026438),
    (0.1495959888165767, 0.6178762444026438),
    (0.1246289712555339, -0.755404408355003),
    (0.1246289712555339, 0.755404408355003),
    (0.0951585116824928, -0.8656312023878318),
    (0.0951585116824928, 0.8656312023878318),
    (0.0622535239386479, -0.9445750230732326),
    (0.0622535239386479, 0.9445750230732326),
    (0.0271524594117541, -0.9894009349916499),
    (0.0271524594117541, 0.9894009349916499),
];

pub const QUADRATURE_8: [(Scalar, Scalar); 8] = [
    (0.362683783378362, -0.1834346424956498),
    (0.362683783378362, 0.1834346424956498),
    (0.3137066458778873, -0.525532409916329),
    (0.3137066458778873, 0.525532409916329),
    (0.2223810344533745, -0.7966664774136267),
    (0.2223810344533745, 0.7966664774136267),
    (0.1012285362903763, -0.9602898564975363),
    (0.1012285362903763, 0.9602898564975363),
];

pub const QUADRATURE_4: [(Scalar, Scalar); 4] = [
    (0.6521451548625461, -0.3399810435848563),
    (0.6521451548625461, 0.3399810435848563),
    (0.3478548451374538, -0.8611363115940526),
    (0.3478548451374538, 0.8611363115940526),
];

/// Find an integral of a function `f` on an interval from `x0` to `x1`
/// using Legendre-Gauss quadrature method.
///
/// This method is equivalent to interpolation of the function with polynomial of degree
/// `table.len() * 2 - 1` and calculating its integral.
///
/// Reference:
///  - [Gauss Quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature)
///  - [Gauss Quadrature](https://www.youtube.com/watch?v=unWguclP-Ds&list=PLC8FC40C714F5E60F)
pub fn integrate_quadrature(
    x0: Scalar,
    x1: Scalar,
    f: impl Fn(Scalar) -> Scalar,
    table: &[(Scalar, Scalar)],
) -> Scalar {
    // quadrature constants are for (-1, 1) interval, we can change variable with
    // `integral(x0, x1, f(x)) = ((b - a) / 2) integral(-1, 1, f((x1 - x0) * x / 2 + (x0 + x1) / 2))`
    let c0 = (x1 - x0) / 2.0;
    let c1 = (x1 + x0) / 2.0;

    let mut result: Scalar = 0.0;
    for (w, x) in table.iter() {
        result += w * f(c0 * x + c1);
    }
    c0 * result
}

/// Check if value is equal to default
/// useful for skipping serialization if value is equal to default value
/// by adding `#[serde(default, skip_serializing_if = "is_default")]`
#[cfg(feature = "serde")]
pub(crate) fn is_default<T: Default + PartialEq>(val: &T) -> bool {
    val == &T::default()
}

/// (De)Serialize with FromStr and Display
/// by adding `#[serde(with = "serde_from_str")]`
#[cfg(feature = "serde")]
pub(crate) mod serde_from_str {
    use serde::{de, Deserialize, Deserializer, Serializer};
    use std::{borrow::Cow, fmt::Display, str::FromStr};

    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Display,
        S: Serializer,
    {
        serializer.collect_str(value)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
    where
        T: FromStr,
        T::Err: Display,
        D: Deserializer<'de>,
    {
        Cow::<'de, str>::deserialize(deserializer)?
            .parse()
            .map_err(de::Error::custom)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[macro_export]
    macro_rules! assert_approx_eq {
        ( $v0:expr, $v1: expr ) => {{
            let (v0, v1) = ($v0, $v1);
            assert!((v0 - v1).abs() < $crate::EPSILON, "{} != {}", v0, v1);
        }};
        ( $v0:expr, $v1: expr, $e: expr ) => {{
            let (v0, v1) = ($v0, $v1);
            assert!((v0 - v1).abs() < $e, "{} != {}", v0, v1);
        }};
    }

    #[macro_export]
    macro_rules! assert_approx_eq_iter {
        ( $v0:expr, $v1: expr ) => {{
            assert_approx_eq_iter!($v0, $v1, $crate::EPSILON);
        }};
        ( $v0:expr, $v1: expr, $e: expr ) => {{
            let mut i0 = $v0.into_iter();
            let mut i1 = $v1.into_iter();
            for (v0, v1) in i0.by_ref().zip(i1.by_ref()) {
                assert_approx_eq!(v0, v1, $e);
            }
            assert!(i0.next().is_none(), "left iterator is longer");
            assert!(i1.next().is_none(), "right iterator is longer");
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
        let mut iter: ArrayIter<u32, 5> = (0..5).collect();
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
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next_back(), None);
    }
}
