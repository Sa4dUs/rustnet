use ndarray::Array2;
use rand::rngs::StdRng;
use crate::lib::math::{ActivationFunction, ErrorFunction};
use crate::neural::layer::layer::NeuralLayer;
use crate::neural::layer::linear::LinearLayer;

pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn NeuralLayer>>
}

impl NeuralNetwork {
    pub fn new(
        in_features: usize,
        topology: Vec<(usize, ActivationFunction)>,
        rng: &mut StdRng,
    ) -> Self {
        let mut layers: Vec<Box<dyn NeuralLayer>> = vec![];
        let mut input_size = in_features;
        for (output_size, act_f) in topology {
            layers.push(Box::new(LinearLayer::new(input_size, output_size, act_f, rng)));
            input_size = output_size
        }

        NeuralNetwork { layers }
    }

    pub fn get_output(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.get_output(&x);
        }
        x
    }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut x = x.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x);
        }
        x
    }

    pub fn backward(&mut self, dy: &Array2<f64>) -> Array2<f64> {
        let mut dy = dy.clone();
        for layer in self.layers.iter_mut().rev() {
            dy = layer.backward(&dy);
        }
        dy
    }

    pub fn train(&mut self, x: &Vec<Array2<f64>>, y: &Vec<Array2<f64>>, learning_rate: f64, loss_f: ErrorFunction) {
        for it in x.iter().zip(y) {
            let (input, expected) = it;
            let actual = self.get_output(&input);

            let loss = loss_f.0(actual.to_owned(), expected.clone());
            let dL_dy = loss_f.1(actual.to_owned(), expected.clone());

            self.backward(&dL_dy);

            for layer in self.layers.iter_mut() {
                layer.stochastic_gradient_descent(learning_rate);
            }
        }
    }
}
