use crate::math::{ActivationFunction, ErrorFunction};
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

        for i in 0..self.layers.len() {
            // MATH DONE
            aux = self.layers[i].forward(out).1;
            out = &aux;
        }

        out.clone()
    }

    pub fn backward(&mut self, x: MatrixF32, y: MatrixF32, cost: ErrorFunction, learning_rate: f32) {
        let mut out = vec![(MatrixF32::new(0, 0), x.clone())];

        for i in 0..self.layers.len() {
            let input = out.last().unwrap().1.clone();
            out.push(self.layers[i].forward(&input));
        }

        let _w: MatrixF32 = MatrixF32::new(0, 0);

        for _ in (0..self.layers.len()).rev() {
            // Step 1: Perform a forward pass and store activations
            let mut activations = vec![x.clone()];
            let mut zs = vec![];

            let mut activation = x.clone();
            for layer in &self.layers {
                let (z, a) = layer.forward(&activation);
                zs.push(z);
                activations.push(a.clone());
                activation = a;
            }

            // Step 2: Calculate the output error
            let mut delta = (cost.1)(activations.last().unwrap().clone(), y.clone())
                .elementwise_mul(zs.last().unwrap().apply(self.layers.last().unwrap().act_f.1));
            let mut deltas = vec![delta.clone()];

            // Step 3: Backpropagate the error
            for l in (1..self.layers.len()).rev() {
                let z = &zs[l - 1];
                let sp = z.apply(self.layers[l - 1].act_f.1);
                delta = (&(&self.layers[l].w.t() * &delta)).elementwise_mul(sp);
                deltas.insert(0, delta.clone());
            }

            // Step 4: Update the weights and biases
            for l in 0..self.layers.len() {
                let delta = &deltas[l];
                let activation = &activations[l];
                self.layers[l].w = &self.layers[l].w - &(&(delta * &activation.t()) * learning_rate);
                self.layers[l].b = &self.layers[l].b - &(&delta.mean_column() * learning_rate);
            }
        }
    }
}