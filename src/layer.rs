use rand::Rng;

use crate::{re_lu, ActivationFunction};

#[derive(Clone, Copy, Debug)]
pub struct Layer<const IN_SIZE: usize, const OUT_SIZE: usize> {
    weights: [[f32; OUT_SIZE]; IN_SIZE],
    biases: [f32; OUT_SIZE],
    act_f: fn(&mut [f32]),
}

impl<const IN_SIZE: usize, const OUT_SIZE: usize> Layer<IN_SIZE, OUT_SIZE> {
    pub fn new(act_f: ActivationFunction) -> Layer<IN_SIZE, OUT_SIZE> {
        let scale: f32 = (2.0 / (IN_SIZE as f32)).sqrt();

        let mut weights: [[f32; OUT_SIZE]; IN_SIZE] = [[0.0; OUT_SIZE]; IN_SIZE];
        (0..IN_SIZE).for_each(|i: usize| {
            (0..OUT_SIZE).for_each(|j: usize| {
                weights[i][j] = (rand::thread_rng().gen::<f32>() - 0.5f32) * 2.0f32 * scale;
            })
        });

        Layer {
            act_f,
            weights,
            biases: [0.0; OUT_SIZE],
        }
    }

    pub fn forward(&self, input: [f32; IN_SIZE]) -> [f32; OUT_SIZE] {
        let mut output: [f32; OUT_SIZE] = self.biases;
        (0..OUT_SIZE).for_each(|i| {
            (0..IN_SIZE).for_each(|j| {
                output[i] += self.weights[j][i] * input[j];
            });
        });

        (self.act_f)(&mut output);
        output
    }

    pub fn backward(
        &mut self,
        input: [f32; IN_SIZE],
        output_grad: [f32; OUT_SIZE],
        alpha: f32,
    ) -> [f32; IN_SIZE] {
        let mut input_grad: [f32; IN_SIZE] = [0.0f32; IN_SIZE];
        (0..OUT_SIZE).for_each(|i| {
            (0..IN_SIZE).for_each(|j| {
                let grad: f32 = output_grad[i] * input[j];
                self.weights[j][i] -= alpha * grad;
                input_grad[j] += output_grad[i] * self.weights[j][i];
            });
            self.biases[i] -= alpha * output_grad[i];
        });

        input_grad
    }
}

impl<const IN_SIZE: usize, const OUT_SIZE: usize> Default for Layer<IN_SIZE, OUT_SIZE> {
    fn default() -> Self {
        Self::new(re_lu)
    }
}
