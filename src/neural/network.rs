use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;
use crate::lib::math::{ActivationFunction, ActivationFunctionsEnum, ErrorFunction, ErrorFunctionsEnum, get_activation_function, get_error_function};
use crate::lib::network_parser::{load_from, save_to};
use crate::neural::layer::layer::NeuralLayer;
use crate::neural::layer::linear::LinearLayer;

pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn NeuralLayer>>
}

impl NeuralNetwork {
    pub fn new_empty() -> Self {
        NeuralNetwork { layers: vec![] }
    }

    pub fn new(
        in_features: usize,
        topology: Vec<(usize, ActivationFunctionsEnum)>,
        rng: &mut StdRng,
    ) -> Self {
        let mut layers: Vec<Box<dyn NeuralLayer>> = vec![];
        let mut input_size = in_features;
        for (output_size, act_f) in topology {
            layers.push(Box::new(LinearLayer::new(input_size, output_size, act_f.as_str(), rng)));
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

    pub fn train(&mut self, x: &Vec<Array2<f64>>, y: &Vec<Array2<f64>>, learning_rate: f64, loss_f: ErrorFunctionsEnum, epochs: usize) {

        let loss_f = get_error_function(loss_f.as_str()).unwrap();
        for _ in 1..epochs {
            for it in x.iter().zip(y) {
                let (input, expected) = it;
                let actual = self.forward(&input);

                let loss = loss_f.0(actual.to_owned(), expected.clone());
                let dL_dy = loss_f.1(actual.to_owned(), expected.clone());

                self.backward(&dL_dy);

                for layer in self.layers.iter_mut() {
                    layer.stochastic_gradient_descent(learning_rate);
                }
            }
        }
    }

    pub fn test(&mut self, x: &Vec<Array2<f64>>, y: &Vec<Array2<f64>>, loss_f: ErrorFunctionsEnum, threshold: f64) {
        let size = x.len();
        let mut hits = 0;
        let loss_f = get_error_function(loss_f.as_str()).unwrap();

        for it in x.iter().zip(y) {
            let (input, expected) = it;
            let actual = self.get_output(&input);

            let loss = loss_f.0(actual.to_owned(), expected.clone());
            println!("Result: {} Expected: {} Loss: {}", actual, expected, loss);

            if loss < threshold {
                hits += 1;
            }
        }

        println!("Testing for {} samples.\nHits: {}\nFails: {}\nHit ratio: {}", size, hits, size - hits, (hits as f32)/(size as f32));
    }

    pub fn save(&self, name: &str){
        let mut parameters: Vec<(Vec<Array2<f64>>, String)> = vec![];
        for layer in &self.layers {
            let (params, act_name) = layer.save();
            parameters.push((params, act_name.to_string()));
        }
        save_to(parameters, name).expect("TODO: panic message");
    }

    pub fn load(&mut self, name: &str, rng: &mut StdRng) {
        let parameters = load_from(name).unwrap();
        self.layers = vec![];
        for params in parameters {
            let input_size = params.0[0].shape()[1];
            let output_size = params.0[0].shape()[0];
            let act_f = &params.1;
            let mut layer = LinearLayer::new(input_size, output_size, act_f, rng);
            layer.load((params.0, &params.1));
            self.layers.push(Box::new(layer));
        }
    }
}
