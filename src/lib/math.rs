use std::any::{type_name, type_name_of_val};
use std::collections::HashMap;
use std::iter::{Iterator, Map};
use std::sync::RwLock;
use ndarray::{Array2, ArrayBase, Axis, OwnedRepr};
use ndarray_rand::rand_distr::num_traits::Pow;
use ndarray_rand::rand_distr::num_traits::real::Real;
use lazy_static::lazy_static;


pub type Function = fn(Array2<f64>) -> Array2<f64>;
pub type ActivationFunction = (Function, Function);
pub type ErrorFunction = (fn(Array2<f64>, Array2<f64>) -> f64, fn(Array2<f64>, Array2<f64>) -> Array2<f64>);

#[derive(Clone, Copy)]
pub enum ActivationFunctionsEnum {
    NULL,
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX
}

impl ActivationFunctionsEnum {
    pub fn as_str(&self) -> &'static str {
        match *self {
            ActivationFunctionsEnum::NULL => "NULL",
            ActivationFunctionsEnum::SIGMOID => "SIGMOID",
            ActivationFunctionsEnum::TANH => "TANH",
            ActivationFunctionsEnum::RELU => "RELU",
            ActivationFunctionsEnum::SOFTMAX => "SOFTMAX"
        }
    }
}


lazy_static! {
    static ref ACTIVATION_FUNCTIONS: RwLock<HashMap<&'static str, ActivationFunction>> = {
        let mut map = HashMap::new();
        map.insert("NULL", (null as Function, null_derivative as Function));
        map.insert("SIGMOID", (sigmoid as Function, sigmoid_derivative as Function));
        map.insert("TANH", (tanh as Function, tanh_derivative as Function));
        map.insert("RELU", (relu as Function, relu_derivative as Function));
        map.insert("SOFTMAX", (softmax as Function, softmax_derivative as Function));
        RwLock::new(map)
    };
}

fn null(t: Array2<f64>) -> Array2<f64> {
    t
}

fn null_derivative(t: Array2<f64>) -> Array2<f64> {
    t.mapv(|_| 1.0)
}

fn sigmoid(t: Array2<f64>) -> Array2<f64> {
    1.0 / (1.0 + t.mapv(|e| (-e).exp()))
}

fn sigmoid_derivative(t: Array2<f64>) -> Array2<f64> {
    (t.mapv(|e| (-e).exp())) / (1.0 + t.mapv(|e| (-e).exp())).mapv(|e| e.powi(2))
}

fn tanh(t: Array2<f64>) -> Array2<f64> {
    (t.mapv(|e| (e).exp()) - t.mapv(|e| (-e).exp())) / (t.mapv(|e| (e).exp()) + t.mapv(|e| (-e).exp()))
}

fn tanh_derivative(t: Array2<f64>) -> Array2<f64> {
    1.0 - tanh(t).mapv(|e| (e).powi(2))
}

fn relu(t: Array2<f64>) -> Array2<f64> {
    t.mapv(|e| f64::max(0.0, e))
}

fn relu_derivative(t: Array2<f64>) -> Array2<f64> {
    t.mapv(|e| if e < 0.0 { 0.0 } else { 1.0 })
}

fn softmax(t: Array2<f64>) -> Array2<f64>
{
    let exp_x = t.mapv(|t| t.exp());
    let sum_exp_x = exp_x.sum();
    exp_x / sum_exp_x
}

// TODO fix dimensions, it may conflict with nn and nl maths
fn softmax_derivative(t: Array2<f64>) -> (Array2<f64>) {
    let softmax_output = softmax(t);

    let (rows, cols) = softmax_output.dim();

    let mut derivative = Array2::zeros((rows, rows));

    for i in 0..rows {
        for j in 0..rows {
            derivative[[i, j]] = softmax_output[[i, 0]] * (((i == j) as usize) as f64 - softmax_output[[i, 0]]);
        }
    }

    derivative
}

pub fn get_activation_function(name: &str) -> Option<ActivationFunction> {
    let map = ACTIVATION_FUNCTIONS.read().unwrap();
    map.get(name).cloned()
}

#[derive(Clone, Copy)]
pub enum ErrorFunctionsEnum {
    MSE,
    SOFTMAX_CROSS_ENTROPY
}

impl ErrorFunctionsEnum {
    pub fn as_str(&self) -> &'static str {
        match *self {
            ErrorFunctionsEnum::MSE => "MSE",
            ErrorFunctionsEnum::SOFTMAX_CROSS_ENTROPY => "SOFTMAX_CROSS_ENTROPY"
        }
    }
}

lazy_static! {
    static ref ERROR_FUNCTIONS: RwLock<HashMap<&'static str, ErrorFunction>> = {
        let mut map = HashMap::new();
        map.insert("MSE", (mse as fn(Array2<f64>, Array2<f64>) -> f64, mse_derivative as fn(Array2<f64>, Array2<f64>) -> Array2<f64>));
        map.insert("SOFTMAX_CROSS_ENTROPY", (softmax_cross_entropy as fn(Array2<f64>, Array2<f64>) -> f64, softmax_cross_entropy_derivative as fn(Array2<f64>, Array2<f64>) -> Array2<f64>));
        RwLock::new(map)
    };
}

fn mse(x: Array2<f64>, y: Array2<f64>) -> f64 {
    (x - y).mapv(|x| x * x).mean().expect("MSE Error")
}

fn mse_derivative(x: Array2<f64>, y: Array2<f64>) -> Array2<f64> {
    2.0 * (x.clone() - y) / x.len() as f64
}

fn softmax_cross_entropy(x: Array2<f64>, y: Array2<f64>) -> f64
{
    let log_likelihood = y * &x.mapv(|t| f64::max(t.ln(), f64::min_value()));
    -log_likelihood.sum() / x.nrows() as f64
}

fn softmax_cross_entropy_derivative(x: Array2<f64>, y: Array2<f64>) -> Array2<f64>
{
    let probs = softmax(x.clone());
    (probs - y) / x.nrows() as f64
}

pub fn get_error_function(name: &str) -> Option<ErrorFunction> {
    let map = ERROR_FUNCTIONS.read().unwrap();
    map.get(name).cloned()
}

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

