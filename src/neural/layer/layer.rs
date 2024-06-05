use ndarray::Array2;
use crate::lib::safe_f64::SafeF64;

pub trait NeuralLayer {
    fn forward(&mut self, x: &Array2<SafeF64>) -> Array2<SafeF64>;
    fn backward(&mut self, dL_dy: &Array2<SafeF64>) -> Array2<SafeF64>;
    fn get_output(&self, x: &Array2<SafeF64>) -> Array2<SafeF64>;
    fn stochastic_gradient_descent(&mut self, learning_rate: SafeF64);
    fn save(&self) -> (Vec<Array2<SafeF64>>, &str);
    fn load(&mut self, parameters: (Vec<Array2<SafeF64>>, &str));
}