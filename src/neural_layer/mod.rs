use std::ops::Mul;
use crate::matrix::MatrixF32;
use crate::math::{ActivationFunction, SIGMOID};

pub struct NeuralLayer {
    pub(crate) w: MatrixF32,
    b: MatrixF32,
    pub(crate) act_f: ActivationFunction
}

impl NeuralLayer {
    pub fn new(n_connections: usize, n_neurons: usize, act_f: ActivationFunction) -> NeuralLayer {
        NeuralLayer {
            w: MatrixF32::new(n_neurons, n_connections).randomized(),
            b  : MatrixF32::new(n_neurons, 1).randomized(),
            act_f
        }
    }

    pub fn forward(&self, x: &MatrixF32) -> (MatrixF32, MatrixF32) {
        // MATH DONE
        let z = &(&self.w.clone()*x) + &self.b.clone();
        let a = z.apply(self.act_f.0);
        (z, a)
    }

    pub fn backward(&mut self, delta: &MatrixF32, prev: &MatrixF32 ,learning_rate: f32) -> MatrixF32 {
        // MATH DONE
        println!("Hola");
        self.w = &self.w - &(&(delta*&prev.t())*learning_rate);
        self.b = &self.b - &(&delta.mean_column() * learning_rate);
        MatrixF32::new(0, 0)
    }
}
