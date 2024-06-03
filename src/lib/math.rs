use std::iter::Iterator;
use ndarray::Array2;
use ndarray_rand::rand_distr::num_traits::Pow;
use ndarray_rand::rand_distr::num_traits::real::Real;

pub type Function = fn(f64) -> f64;
pub type ActivationFunction = (Function, Function);
pub type ErrorFunction = (fn(Array2<f64>, Array2<f64>) -> f64, fn(Array2<f64>, Array2<f64>) -> Array2<f64>);
pub const NULL: ActivationFunction = (|t| t, |t| 1.0);
pub const SIGMOID: ActivationFunction = (|t| 1.0/(1.0+(-t).exp()), |t| ((-t).exp())/(1.0 + (-t).exp()).pow(2));
pub const TANH: ActivationFunction = (|t| (t.exp() - (-t).exp())/(t.exp() + (-t).exp()), |t| 1.0 - TANH.0(t).powi(2));
pub const RELU: ActivationFunction = (|t| f64::max(0.0, t), |t| return if t < 0.0 { 0.0 } else { 1.0 });

pub const MSE: ErrorFunction = (|x, y| (x - y).mapv(|x| x * x).mean().expect("MSE Error"), |x, y| 2.0 * (x.clone() - y) / x.len() as f64);
pub const CROSS_ENTROPY: ErrorFunction = (|x, y| (-1.0/x.len() as f64)*(x.iter().zip(y).map(|(xi, yi)| yi*(xi.ln())+(1.0-yi)*((1.0-xi).ln())).sum::<f64>()), |x, y| y.clone()/x.clone() + (1.0-y)/(1.0-x));

pub fn mean(a: &Array2<f64>) -> f64 {
    a.sum() / (a.len() as f64)
}

pub fn probability_density_function(x: f64, y: f64) -> f64 {
    const SDX: f64 = 0.1;
    const SDY: f64 = 0.1;
    const A: f64 = 5.0;
    let x = x / 10.0;
    let y = y / 10.0;

    A * (-x * x / 2.0 / SDX / SDX - y * y / 2.0 / SDY / SDY).exp()
}
