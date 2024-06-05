use std::fmt;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;
use crate::lib::math::{ActivationFunction, ActivationFunctionsEnum, get_activation_function};
use crate::lib::safe_f64::{SafeF64, Dot};
use crate::neural::layer::layer::NeuralLayer;

pub struct LinearLayer {
    pub W: Array2<SafeF64>,
    pub b: Array2<SafeF64>,

    pub dL_dW: Option<Array2<SafeF64>>,
    pub dL_db: Option<Array2<SafeF64>>,

    pub dy_dW: Option<Array2<SafeF64>>,
    pub act_f: ActivationFunction,

    act_name: String
}

impl NeuralLayer for LinearLayer {
    fn forward(&mut self, x: &Array2<SafeF64>) -> Array2<SafeF64> {
        self.dy_dW = Some(x.to_owned());
        self.get_output(x)
    }

    fn backward(&mut self, dL_dy: &Array2<SafeF64>) -> Array2<SafeF64> {
        let dL_dW = dL_dy.t().dot(
            &self
                .dy_dW
                .as_ref()
                .expect("Need to call forward() first.")
                .view(),
        );

        let dL_db = dL_dy.t().dot(&Array2::ones((dL_dy.shape()[0], 1)));

        self.dL_dW = Some(dL_dW);
        self.dL_db = Some(dL_db.to_owned());

        dL_dy.dot(&self.W)
    }

    fn get_output(&self, x: &Array2<SafeF64>) -> Array2<SafeF64> {
        self.act_f.0((self.W.dot(&x.t()) + self.b.clone()).t().to_owned())
    }

    fn stochastic_gradient_descent(&mut self, learning_rate: SafeF64) {
        println!("Before W: {}, B: {}", self.W, self.b);

        self.W =
            self.W.clone() - self.dL_dW.clone().expect("No gradient registered") * learning_rate;
        self.b =
            self.b.clone() - self.dL_db.clone().expect("No gradient registered") * learning_rate;

        println!("After W: {}, B: {}", self.W, self.b);

    }

    fn save(&self) -> (Vec<Array2<SafeF64>>, &str) {
        (vec![self.W.clone(), self.b.clone()], self.act_name.as_str())
    }

    fn load(&mut self, parameters: (Vec<Array2<SafeF64>>, &str)) {
        self.W = parameters.0[0].clone();
        self.b = parameters.0[1].clone()
    }
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize, act_f: &str, rng: &mut StdRng) -> Self {
        let W = Array2::from_shape_fn((output_size, input_size), |_| SafeF64::new(rng.gen::<f64>()));
        let b = Array2::from_shape_fn((output_size, 1), |_| SafeF64::new(rng.gen::<f64>()));
        let act_name: String = act_f.to_owned();
        let act_f = get_activation_function(act_f).unwrap();
        LinearLayer {
            W,
            b,
            dL_dW: None,
            dL_db: None,
            dy_dW: None,
            act_f,
            act_name
        }
    }
}

impl fmt::Display for LinearLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "W: {}, B: {}", self.W, self.b)
    }
}