use std::cmp::Ordering;
use std::fmt;
pub use ndarray::{Array2, ArrayBase, Axis, DataMut, Ix2};
use std::ops::{Add, Div, Mul, Sub, Neg};
use ndarray::{OwnedRepr, ViewRepr};
use ndarray_rand::rand_distr::num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use ndarray_rand::rand_distr::num_traits::real::Real;
use serde::{Deserialize, Deserializer, Serialize, Serializer};


#[derive(Debug, Clone, Copy)]
pub struct SafeF64(pub(crate) f64);

impl fmt::Display for SafeF64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq for SafeF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}


impl PartialOrd for SafeF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Serialize for SafeF64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SafeF64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Ok(SafeF64::new(value))
    }
}

impl SafeF64 {

    pub fn new(value: f64) -> Self {
        if value.is_infinite() {
            SafeF64(f64::MAX.copysign(value))
        } else {
            SafeF64(value)
        }
    }

    pub fn exp(self) -> Self {
        SafeF64::new(self.0.exp())
    }

    pub fn ln(self) -> Self {
        SafeF64::new(self.0.ln())
    }

    pub fn powi(self, n: i32) -> Self {
        SafeF64::new(self.0.powi(n))
    }
}

// Implementación de las operaciones aritméticas para SafeF64
impl Add for SafeF64 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let result = self.0 + other.0;
        if result.is_infinite() {
            SafeF64(f64::MAX.copysign(result))
        } else {
            SafeF64(result)
        }
    }
}

impl Sub for SafeF64 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let result = self.0 - other.0;
        if result.is_infinite() {
            SafeF64(f64::MAX.copysign(result))
        } else {
            SafeF64(result)
        }
    }
}

impl Mul for SafeF64 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let result = self.0 * other.0;
        if result.is_infinite() {
            SafeF64(f64::MAX.copysign(result))
        } else {
            SafeF64(result)
        }
    }
}

impl Div for SafeF64 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let result = self.0 / other.0;
        if result.is_infinite() || other.0 == 0.0 {
            SafeF64(f64::MAX.copysign(result))
        } else {
            SafeF64(result)
        }
    }
}

impl Neg for SafeF64 {
    type Output = Self;

    fn neg(self) -> Self {
        SafeF64::new(-self.0)
    }
}

// Implementación de operaciones entre SafeF64 y Array2<SafeF64>
impl Add<Array2<SafeF64>> for SafeF64 {
    type Output = Array2<SafeF64>;

    fn add(self, rhs: Array2<SafeF64>) -> Array2<SafeF64> {
        rhs.mapv(|elem| self + elem)
    }
}

impl Sub<Array2<SafeF64>> for SafeF64 {
    type Output = Array2<SafeF64>;

    fn sub(self, rhs: Array2<SafeF64>) -> Array2<SafeF64> {
        rhs.mapv(|elem| self - elem)
    }
}

impl Mul<Array2<SafeF64>> for SafeF64 {
    type Output = Array2<SafeF64>;

    fn mul(self, rhs: Array2<SafeF64>) -> Array2<SafeF64> {
        rhs.mapv(|elem| self * elem)
    }
}

impl Div<Array2<SafeF64>> for SafeF64 {
    type Output = Array2<SafeF64>;

    fn div(self, rhs: Array2<SafeF64>) -> Array2<SafeF64> {
        rhs.mapv(|elem| self / elem)
    }
}

// Implementación de operaciones entre Array2<SafeF64> y SafeF64
impl Add<SafeF64> for Array2<SafeF64> {
    type Output = Array2<SafeF64>;

    fn add(self, rhs: SafeF64) -> Array2<SafeF64> {
        self.mapv(|elem| elem + rhs)
    }
}

impl Sub<SafeF64> for Array2<SafeF64> {
    type Output = Array2<SafeF64>;

    fn sub(self, rhs: SafeF64) -> Array2<SafeF64> {
        self.mapv(|elem| elem - rhs)
    }
}

impl Mul<SafeF64> for Array2<SafeF64> {
    type Output = Array2<SafeF64>;

    fn mul(self, rhs: SafeF64) -> Array2<SafeF64> {
        self.mapv(|elem| elem * rhs)
    }
}

impl Div<SafeF64> for Array2<SafeF64> {
    type Output = Array2<SafeF64>;

    fn div(self, rhs: SafeF64) -> Array2<SafeF64> {
        self.mapv(|elem| elem / rhs)
    }
}

// Implementación del trait Zero para SafeF64
impl Zero for SafeF64 {
    fn zero() -> Self {
        SafeF64::new(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for SafeF64 {
    fn one() -> Self {
        SafeF64::new(1.0)
    }
}

// Implementación del trait FromPrimitive para SafeF64
impl FromPrimitive for SafeF64 {
    fn from_i64(n: i64) -> Option<Self> {
        Some(SafeF64::new(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(SafeF64::new(n as f64))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(SafeF64::new(n))
    }
}

impl ToPrimitive for SafeF64 {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 as i64)
    }

    fn to_usize(&self) -> Option<usize> {
        Some(self.0 as usize)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }
}

pub trait Dot<Rhs = Self> {
    type Output;
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

impl<'a> Dot<ArrayBase<OwnedRepr<SafeF64>, Ix2>> for ArrayBase<ViewRepr<&'a SafeF64>, Ix2> {
    type Output = Array2<SafeF64>;

    fn dot(&self, rhs: &ArrayBase<OwnedRepr<SafeF64>, Ix2>) -> Self::Output {
        let mut result = Array2::<SafeF64>::zeros((self.dim().0, rhs.dim().1));
        for i in 0..self.dim().0 {
            for j in 0..rhs.dim().1 {
                for k in 0..self.dim().1 {
                    result[[i, j]] = result[[i, j]] + self[[i, k]] * rhs[[k, j]];
                }
            }
        }
        result
    }
}

impl<'a, 'b> Dot<&'b ArrayBase<OwnedRepr<SafeF64>, Ix2>> for ArrayBase<ViewRepr<&'a SafeF64>, Ix2> {
    type Output = Array2<SafeF64>;

    fn dot(&self, rhs: &&ArrayBase<OwnedRepr<SafeF64>, Ix2>) -> Self::Output {
        self.dot(*rhs)
    }
}

impl<'a, 'b> Dot<ArrayBase<ViewRepr<&'b SafeF64>, Ix2>> for ArrayBase<ViewRepr<&'a SafeF64>, Ix2> {
    type Output = Array2<SafeF64>;

    fn dot(&self, rhs: &ArrayBase<ViewRepr<&'b SafeF64>, Ix2>) -> Self::Output {
        let mut result = Array2::<SafeF64>::zeros((self.dim().0, rhs.dim().1));
        for i in 0..self.dim().0 {
            for j in 0..rhs.dim().1 {
                for k in 0..self.dim().1 {
                    result[[i, j]] = result[[i, j]] + self[[i, k]] * rhs[[k, j]];
                }
            }
        }
        result
    }
}

