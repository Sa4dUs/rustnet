use std::ops::Mul;
use crate::matrix::MatrixF32;
use crate::math::{ActivationFunction, SIGMOID};

pub struct NeuralLayer {
    w: MatrixF32,
    b: MatrixF32,
    act_f: ActivationFunction
}

impl NeuralLayer {
    pub fn new(n_connections: usize, n_neurons: usize, act_f: ActivationFunction) -> NeuralLayer {
        NeuralLayer {
            w: MatrixF32::new(n_connections, n_neurons).randomized(),
            b: MatrixF32::new(n_neurons, 1).randomized(),
            act_f
        }
    }

    pub fn forward(&self, x: &MatrixF32) -> (MatrixF32, MatrixF32) {
        let z = &(&self.w.clone().t()*x) + &self.b.clone();
        let a = z.apply(self.act_f.0);
        (z, a)
    }

    pub fn backward(&mut self, delta: MatrixF32, prev: MatrixF32 ,learning_rate: f32) -> MatrixF32 {
        self.w = &self.w - &(&(&prev*&delta)*learning_rate);
        self.b = &self.b - &delta;
        MatrixF32::new(0, 0)
    }
}
