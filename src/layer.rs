use crate::{activation::ActivationFunction, Layer};
use rand::Rng;

impl Layer {
    /// Creates a new `Layer` instance with randomly initialized weights and biases.
    /// The weights are initialized using a random distribution scaled by the inverse square root
    /// of the input size for better convergence during training.
    ///
    /// # Arguments
    /// * `in_size` - The number of input features to this layer (i.e., the size of the input vector).
    /// * `out_size` - The number of output features produced by this layer (i.e., the size of the output vector).
    /// * `act_f` - The activation function to apply to the output of this layer.
    ///
    /// # Returns
    /// A new `Layer` instance with initialized weights, biases, and the given activation function.
    ///
    /// # Example
    /// ```
    /// let layer = Layer::new(10, 5, activation::relu);
    /// assert_eq!(layer.input_size, 10);
    /// assert_eq!(layer.output_size, 5);
    /// ```
    pub fn new(in_size: usize, out_size: usize, act_f: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        // Scaling factor for weight initialization, based on the input size
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
            activation_function: act_f,
        }
    }

    /// Performs a forward pass through the layer.
    /// It computes the dot product of the input vector with the layer's weights and adds the biases.
    /// The result is then passed through the activation function.
    ///
    /// # Arguments
    /// * `input` - A slice of `f32` values representing the input to the layer.
    ///
    /// # Returns
    /// A `Vec<f32>` representing the output of the layer after applying the activation function.
    ///
    /// # Example
    /// ```
    /// let layer = Layer::new(3, 2, activation::relu);
    /// let input = vec![0.5, 0.3, -0.2];
    /// let output = layer.forward(&input);
    /// assert_eq!(output.len(), 2);
    /// ```
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_size];

        (0..self.output_size).for_each(|i| {
            output[i] = self.biases[i];

            (0..self.input_size).for_each(|j| {
                output[i] += input[j] * self.weights[j * self.output_size + i];
            });
        });

        (self.activation_function)(&output)
    }

    /// Performs a backward pass through the layer.
    /// It computes the gradient of the loss with respect to the weights, biases, and input.
    /// This method updates the weights and biases using gradient descent.
    ///
    /// # Arguments
    /// * `input` - A slice of `f32` values representing the input to the layer.
    /// * `output_grad` - A slice of `f32` values representing the gradient of the loss with respect to the output.
    /// * `lr` - A `f32` value representing the learning rate used to update the weights and biases.
    ///
    /// # Returns
    /// A `Vec<f32>` representing the gradient of the loss with respect to the input of this layer.
    ///
    /// # Example
    /// ```
    /// let mut layer = Layer::new(3, 2, activation::relu);
    /// let input = vec![0.5, 0.3, -0.2];
    /// let output_grad = vec![0.1, -0.1];
    /// let lr = 0.01;
    /// let input_grad = layer.backward(&input, &output_grad, lr);
    /// assert_eq!(input_grad.len(), 3);
    /// ```
    pub fn backward(&mut self, input: &[f32], output_grad: &[f32], lr: f32) -> Vec<f32> {
        let mut input_grad = vec![0.0; self.input_size];

        (0..self.output_size).for_each(|i| {
            for j in 0..self.input_size {
                let idx = j * self.output_size + i;
                let grad = output_grad[i] * input[j];
                self.weights[idx] -= lr * grad;
                input_grad[j] += output_grad[i] * self.weights[idx];
            }
            self.biases[i] -= lr * output_grad[i];
        });

        input_grad
    }
}
