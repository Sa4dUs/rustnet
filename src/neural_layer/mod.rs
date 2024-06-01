use std::ops::Mul;
use crate::matrix::MatrixF32;
use crate::math::{ActivationFunction, SIGMOID};

pub struct NeuralLayer {
    pub(crate) w: MatrixF32,
    pub(crate) b: MatrixF32,
    pub(crate) act_f: ActivationFunction
}

impl NeuralLayer {
    pub fn new(n_connections: usize, n_neurons: usize, act_f: ActivationFunction) -> NeuralLayer {
        NeuralLayer {
            w: MatrixF32::new(n_neurons, n_connections).randomized(5.0),
            b  : MatrixF32::new(n_neurons, 1).randomized(5.0),
            act_f
        }
    }

    pub fn forward(&self, x: &MatrixF32) -> (MatrixF32, MatrixF32) {
        // MATH DONE
        let z = &(&self.w.clone()*x) + &self.b.clone();
        let a = z.apply(self.act_f.0);
        (z, a)
    }
}
