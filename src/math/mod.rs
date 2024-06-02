use std::cmp::min;
use crate::matrix::MatrixF32;

pub type Function = fn(f32) -> f32;
pub type ActivationFunction = (Function, Function);
pub type ErrorFunction = (fn(MatrixF32, MatrixF32) -> f32, fn(MatrixF32, MatrixF32) -> MatrixF32);

pub const SIGMOID: ActivationFunction = (|t| 1.0/(1.0+(-t).exp()), |t| SIGMOID.0(t)*(1.0-SIGMOID.0(t)));
pub const RELU: ActivationFunction = (|t| f32::max(0.0, t), |t| return if t < 0.0 { 0.0 } else { 1.0 });
pub const MSE: ErrorFunction = (|x, y| mean((&(&x - &y)).elementwise_mul(&(&x - &y))), |x, y| &(&x - &y) * (2.0f32/(x.get_rows() as f32)));

pub fn mean(a: MatrixF32) -> f32 {
    let mut sum: f32 = 0.0;

    let rows = a.get_rows();
    let cols = a.get_cols();

    for row in 0..rows {
        for col in 0..cols {
            sum += a.get(row, col);
        }
    }

    sum/((rows*cols) as f32)
}