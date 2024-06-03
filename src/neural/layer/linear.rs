use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;
use crate::lib::math::{ActivationFunction, RELU};

use crate::neural::layer::layer::NeuralLayer;

pub struct LinearLayer {
    pub W: Array2<f64>,
    pub b: Array2<f64>,

    pub dL_dW: Option<Array2<f64>>,
    pub dL_db: Option<Array2<f64>>,

    pub dy_dW: Option<Array2<f64>>,
    pub act_f: ActivationFunction,
}

impl NeuralLayer for LinearLayer {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.dy_dW = Some(x.to_owned());
        self.get_output(x)
    }

    fn backward(&mut self, dL_dy: &Array2<f64>) -> Array2<f64> {
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

    fn get_output(&self, x: &Array2<f64>) -> Array2<f64> {
        (self.W.dot(&x.t()) + self.b.clone()).t().to_owned().mapv(self.act_f.0)
    }

    fn stochastic_gradient_descent(&mut self, learning_rate: f64) {
        self.W =
            self.W.clone() - self.dL_dW.clone().expect("No gradient registered") * learning_rate;
        self.b =
            self.b.clone() - self.dL_db.clone().expect("No gradient registered") * learning_rate;
    }
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize, act_f: ActivationFunction, rng: &mut StdRng) -> Self {
        let W = Array2::from_shape_fn((output_size, input_size), |_| rng.gen::<f64>());
        let b = Array2::from_shape_fn((output_size, 1), |_| rng.gen::<f64>());

        LinearLayer {
            W,
            b,
            dL_dW: None,
            dL_db: None,
            dy_dW: None,
            act_f
        }
    }
}
