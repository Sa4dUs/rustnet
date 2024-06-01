use crate::math::{ActivationFunction, ErrorFunction, SIGMOID};
use crate::matrix::MatrixF32;
use crate::neural_layer::NeuralLayer;

pub struct NeuralNetwork {
    layers: Vec<NeuralLayer>
}

impl NeuralNetwork {
    pub fn new(topology: &Vec<(usize, ActivationFunction)>) -> NeuralNetwork {
        NeuralNetwork {
            layers: (1..topology.len()).collect::<Vec<usize>>().iter().map(|t| NeuralLayer::new(topology[*t - 1].0, topology[*t].0, topology[*t].1)).collect()
        }
    }

    pub fn forward(&self, x: &MatrixF32) -> MatrixF32 {
        let mut out: &MatrixF32 = x;
        let mut aux: MatrixF32;

        for i in (0..self.layers.len()) {
            // MATH DONE
            aux = self.layers[i].forward(out).1;
            out = &aux;
        }

        out.clone()
    }

    pub fn backward(&mut self, x: MatrixF32, y: MatrixF32, cost: ErrorFunction, learning_rate: f32) {
        let mut out = vec![(MatrixF32::new(0, 0), x.clone())];

        // Forward pass
        for i in 0..self.layers.len() {
            let input = out.last().unwrap().1.clone();
            out.push(self.layers[i].forward(&input));
        }

        // Backward pass + Gradient Descent
        let mut deltas = vec![];
        let mut _w: MatrixF32 = MatrixF32::new(0, 0);

        for l in (0..self.layers.len()).rev() {
            let (a, z) = out[l+1].clone();

            if l == self.layers.len() - 1 {
                deltas.insert(0, MatrixF32::new(self.layers[l].w.get_rows(), 1));
            } else {
                // Pending delta when l != L
                deltas.insert(0, MatrixF32::new(self.layers[l].w.get_rows(), 1));
            }

            _w = self.layers[l].w.clone();

            &self.layers[l].backward(&deltas[0], &out[l].1, learning_rate);
        }
    }
}