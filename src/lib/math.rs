use std::iter::Iterator;
use ndarray::{Array2, Axis};
use ndarray_rand::rand_distr::num_traits::Pow;
use ndarray_rand::rand_distr::num_traits::real::Real;

pub type Function = fn(Array2<f64>) -> Array2<f64>;
pub type ActivationFunction = (Function, Function);
pub type ErrorFunction = (fn(Array2<f64>, Array2<f64>) -> f64, fn(Array2<f64>, Array2<f64>) -> Array2<f64>);
pub const NULL: ActivationFunction = (|t| t, |t| t.mapv(|_| 1.0));
pub const SIGMOID: ActivationFunction = (|t| 1.0/(1.0+t.mapv(|e| (-e).exp())), |t| (t.mapv(|e| (-e).exp()))/(1.0 + t.mapv(|e| (-e).exp())).mapv(|e| e.powi(2)));
pub const TANH: ActivationFunction = (|t| (t.mapv(|e| (e).exp()) - t.mapv(|e| (-e).exp()))/(t.mapv(|e| (e).exp()) + t.mapv(|e| (-e).exp())), |t| 1.0 - TANH.0(t).mapv(|e| (e).powi(2)));
pub const RELU: ActivationFunction = (|t| t.mapv(|e| f64::max(0.0, e)), |t| t.mapv(|e| return if e < 0.0 { 0.0 } else { 1.0 }));

pub const MSE: ErrorFunction = (|x, y| (x - y).mapv(|x| x * x).mean().expect("MSE Error"), |x, y| 2.0 * (x.clone() - y) / x.len() as f64);
pub const CROSS_ENTROPY: ErrorFunction = (|x, y| (-1.0/x.len() as f64)*(x.iter().zip(y).map(|(xi, yi)| yi*(xi.ln())+(1.0-yi)*((1.0-xi).ln())).sum::<f64>()), |x, y| y.clone()/x.clone() + (1.0-y)/(1.0-x));
pub const SOFTMAX_CROSS_ENTROPY: ErrorFunction = (
    |x: Array2<f64>, y: Array2<f64>| -> f64 {
        let softmax = |row: &Array2<f64>| -> Array2<f64> {
            let max_x = row.map_axis(Axis(1), |x| *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
            let exp_x = row - &max_x;
            let exp_x = exp_x.mapv(|xi| xi.exp());
            let sum_exp_x = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));
            exp_x / sum_exp_x
        };

        let probs = softmax(&x);
        let log_likelihood = y * &probs.mapv(f64::ln);
        -log_likelihood.sum() / x.nrows() as f64
    },
    |x: Array2<f64>, y: Array2<f64>| -> Array2<f64> {
        let softmax = |row: &Array2<f64>| -> Array2<f64> {
            let max_x = row.map_axis(Axis(1), |x| *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap());
            let exp_x = row - &max_x;
            let exp_x = exp_x.mapv(|xi| xi.exp());
            let sum_exp_x = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));
            exp_x / sum_exp_x
        };

        let probs = softmax(&x);
        (probs - y) / x.nrows() as f64
    }
);

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
