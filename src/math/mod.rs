use crate::matrix::MatrixF32;

pub type Function = fn(f32) -> f32;
pub type ActivationFunction = (Function, Function);
pub type ErrorFunction = (fn(MatrixF32, MatrixF32) -> f32, fn(MatrixF32, MatrixF32) -> MatrixF32);

pub const SIGMOID: ActivationFunction = (|t| 1.0/(1.0-(-t).exp()), |t| t*(1.0-t));
pub const MSE: ErrorFunction = (|x, y| mean((&(&x - &y)).elementwise_mul(&x - &y)), |x, y| &x - &y);

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

pub fn mse(result: &MatrixF32, expected: &MatrixF32) -> f32 {
    assert_eq!(result.get_cols(), 1, "Lost functions can only be computed for vectors");
    assert_eq!(expected.get_cols(), 1, "Lost functions can only be computed for vectors");

    mean(&((result-expected).apply(|x:f32|x.powf(2.0))))
}