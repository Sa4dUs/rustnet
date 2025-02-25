use crate::{loss::LossFunction, Layer, NeuralNetwork};

impl<'a> NeuralNetwork<'a> {
    /// Creates a new instance of the `NeuralNetwork`.
    /// This method initializes the network with the provided layers and performs basic validation.
    ///
    /// # Arguments
    /// * `layers` - A mutable slice of `Layer` instances that defines the structure of the neural network.
    ///
    /// # Returns
    /// A new `NeuralNetwork` instance.
    ///
    /// # Panics
    /// This method will panic if there are no layers provided (i.e., if the slice is empty).
    ///
    /// # Example
    /// ```
    /// let layers = vec![
    ///     Layer::new(3, 5, activation::relu),
    ///     Layer::new(5, 2, activation::softmax),
    /// ];
    /// let mut nn = NeuralNetwork::new(&mut layers);
    /// assert_eq!(nn.input_size, 3);
    /// assert_eq!(nn.output_size, 2);
    /// ```
    pub fn new(layers: &'a mut [Layer]) -> Self {
        let n: usize = layers.len();
        assert!(n > 0, "NeuralNetwork must have at least one layer.");

        NeuralNetwork {
            input_size: layers[0].input_size,
            output_size: layers[n - 1].output_size,
            layers,
        }
    }

    /// Performs a forward pass through the entire neural network.
    /// The method iteratively passes the input through each layer of the network,
    /// returning the output of the last layer.
    ///
    /// # Arguments
    /// * `input` - A slice of `f32` representing the input data to the network.
    ///
    /// # Returns
    /// A `Vec<f32>` representing the final output of the network after passing through all layers.
    ///
    /// # Panics
    /// This method will panic if the input size does not match the expected input size of the network.
    ///
    /// # Example
    /// ```
    /// let layers = vec![
    ///     Layer::new(3, 5, activation::relu),
    ///     Layer::new(5, 2, activation::softmax),
    /// ];
    /// let mut nn = NeuralNetwork::new(&mut layers);
    /// let input = vec![0.5, -0.2, 1.1];
    /// let output = nn.forward(&input);
    /// assert_eq!(output.len(), 2); // Output size matches the final layer's output size
    /// ```
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_size, "Input size mismatch.");

        let n = self.layers.len();
        let mut buffer1 = input.to_vec();
        let mut buffer2 = vec![0.0; self.output_size];

        let mut input_buffer = &mut buffer1;
        let mut output_buffer = &mut buffer2;

        for (i, layer) in self.layers.iter().enumerate() {
            if i < n - 1 {
                output_buffer.resize(layer.output_size, 0.0);
            }

            *output_buffer = layer.forward(input_buffer);
            std::mem::swap(&mut input_buffer, &mut output_buffer);
        }

        input_buffer.clone()
    }

    /// Trains the neural network using the provided training data, target values, and loss function.
    /// The method performs backpropagation to update the weights and biases of each layer,
    /// minimizing the loss function over multiple epochs.
    ///
    /// # Arguments
    /// * `inputs` - A slice of `Vec<f32>` representing the input data for training.
    /// * `targets` - A slice of `Vec<f32>` representing the target labels corresponding to the inputs.
    /// * `learning_rate` - A `f32` value representing the learning rate used for gradient descent.
    /// * `loss_function` - A `LossFunction` instance that computes the loss and its gradient.
    /// * `epochs` - The number of training epochs (iterations over the entire dataset).
    ///
    /// # Example
    /// ```
    /// let layers = vec![
    ///     Layer::new(3, 5, activation::relu),
    ///     Layer::new(5, 2, activation::softmax),
    /// ];
    /// let mut nn = NeuralNetwork::new(&mut layers);
    ///
    /// let inputs = vec![
    ///     vec![0.5, -0.2, 1.1],
    ///     vec![0.9, 0.4, -0.1],
    /// ];
    /// let targets = vec![
    ///     vec![1.0, 0.0],
    ///     vec![0.0, 1.0],
    /// ];
    ///
    /// nn.train(&inputs, &targets, 0.01, loss::MSE, 1000);
    /// ```
    pub fn train(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        learning_rate: f32,
        loss_function: LossFunction,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                let mut activations: Vec<Vec<f32>> = Vec::new();

                let mut temp_input = input.clone();
                activations.push(temp_input.clone());

                for layer in &mut *self.layers {
                    temp_input = layer.forward(&temp_input);
                    activations.push(temp_input.clone());
                }

                total_loss += (loss_function.function)(activations.last().unwrap(), target);

                let mut grad = (loss_function.derivative)(activations.last().unwrap(), target);

                for i in (0..self.layers.len()).rev() {
                    grad = self.layers[i].backward(&activations[i], &grad, learning_rate);
                }
            }

            let average_loss = total_loss / inputs.len() as f32;
            println!("Epoch {}: Loss = {}", epoch + 1, average_loss);
        }
    }
}
