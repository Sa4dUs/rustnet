/// The `activation` module defines various activation functions used in the neural network.
pub mod activation;
/// The `loss` module contains loss functions used for training the neural network.
pub mod loss;

mod layer;
mod neural_network;

use activation::ActivationFunction;

/// `Layer` represents a single layer in the neural network.
/// It contains weights, biases, and information about the size of the layer.
/// It also contains an activation function used to transform the output of this layer.
pub struct Layer {
    /// A vector holding the weights for the neurons in this layer.
    /// The length of the vector is determined by the input and output size of the layer.
    weights: Vec<f32>,

    /// A vector holding the biases for the neurons in this layer.
    /// The length of the vector is equal to the output size of the layer.
    biases: Vec<f32>,

    /// The number of input features (neurons) for this layer.
    /// It must match the output size of the previous layer.
    pub input_size: usize,

    /// The number of output features (neurons) produced by this layer.
    /// It determines the size of the output vector for this layer.
    pub output_size: usize,

    /// The activation function applied to the output of this layer.
    /// This function defines how the layer's output is transformed before passing to the next layer.
    activation_function: ActivationFunction,
}

/// `NeuralNetwork` represents a full feedforward neural network with multiple layers.
/// It contains information about the input size, output size, and the layers that make up the network.
pub struct NeuralNetwork<'a> {
    /// The number of input features for the neural network.
    /// This must match the size of the input data provided during training or inference.
    input_size: usize,

    /// The number of output features for the neural network.
    /// This determines the dimensionality of the network's final output.
    output_size: usize,

    /// A mutable reference to an array of `Layer` instances that make up the neural network.
    /// Each layer in the network has weights, biases, and an activation function.
    layers: &'a mut [Layer],
}

#[cfg(test)]
/// A test module for testing the functionality of the neural network crate.
/// This module includes unit tests that verify the correctness of the crate's functionality.
mod tests {
    /// A trivial test that always passes, used as a basic sanity check.
    /// This is typically used to verify that the testing framework is set up correctly.
    #[test]
    fn test_trivial() {
        // The test asserts that 1 equals 1, which is trivially true.
        assert_eq!(1, 1);
    }
}
