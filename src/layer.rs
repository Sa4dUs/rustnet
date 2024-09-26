use rand::Rng;
pub struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl Layer {
    pub fn new(in_size: usize, out_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / in_size as f32).sqrt();
        let weights: Vec<f32> = (0..in_size * out_size)
            .map(|_| ((rng.gen::<f32>() - 0.5) * 2.0) * scale)
            .collect();
        let biases: Vec<f32> = vec![0.0; out_size];

        Layer {
            weights,
            biases,
            input_size: in_size,
            output_size: out_size,
        }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        (0..self.output_size).for_each(|i| {
            output[i] = self.biases[i];

            (0..self.input_size).for_each(|j| {
                output[i] += input[j] * self.weights[j * self.output_size + i];
            })
        });
    }

    pub fn backward(
        &mut self,
        input: &[f32],
        output_grad: &[f32],
        mut input_grad: Option<&mut [f32]>,
        lr: f32,
    ) {
        (0..self.output_size).for_each(|i| {
            for j in 0..self.input_size {
                let idx = j * self.output_size + i;
                let grad = output_grad[i] * input[j];
                self.weights[idx] -= lr * grad;

                if let Some(input_grad) = &mut input_grad {
                    input_grad[j] += output_grad[i] * self.weights[idx];
                }
            }

            self.biases[i] -= lr * output_grad[i];
        });
    }
}
