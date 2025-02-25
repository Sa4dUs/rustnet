# Scorch

Scorch is a straightforward , easy-to-use neural network library written in Rust. It is designed for educational purposes and quick experiments with neural networks, providing basic building blocks for building, training, and evaluating feedforward neural networks.

## Features

-   **Feedforward Neural Networks**: Build and train fully connected networks.
-   **Activation Functions**: Supports popular activation functions like ReLU, Sigmoid, and Tanh.
-   **Training Algorithms**: Implements basic gradient descent-based training with backpropagation.
-   **Modular Design**: Easily extendable to include more advanced features or custom components.

## Installation

To use `scorch` in your Rust project, add it as a dependency in your `Cargo.toml`:

```toml
[dependencies]
scorch = "0.1"
```

## Usage

Here's a simple example of how to create and train a neural network with `scorch`.

### Example: Basic Neural Network

```rust
use crate::{Layer, activation::{relu, sigmoid}, loss::{MSE}, NeuralNetwork};

fn main() {
    // Define the layers of the neural network.
    let mut layers = vec![
        Layer::new(2, 4, relu),   // First layer with 2 input features and 4 output features (hidden layer)
        Layer::new(4, 1, sigmoid), // Second layer with 4 input features and 1 output feature (output layer)
    ];

    // Create the neural network
    let mut nn = NeuralNetwork::new(&mut layers);

    // Define the training data for XOR
    let inputs = vec![
        vec![0.0, 0.0], // XOR input: (0, 0)
        vec![0.0, 1.0], // XOR input: (0, 1)
        vec![1.0, 0.0], // XOR input: (1, 0)
        vec![1.0, 1.0], // XOR input: (1, 1)
    ];

    let targets = vec![
        vec![0.0], // XOR output: 0
        vec![1.0], // XOR output: 1
        vec![1.0], // XOR output: 1
        vec![0.0], // XOR output: 0
    ];

    // Train the neural network with XOR data
    nn.train(&inputs, &targets, 0.1, MSE, 10_000);  // Using a learning rate of 0.1 and training for 10,000 epochs

    // Test the trained model with the XOR inputs
    let test_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    for test_input in test_inputs {
        let prediction = nn.forward(&test_input);
        println!("Input: {:?}, Predicted Output: {:?}", test_input, prediction);
    }
}
```

### How It Works

-   **Layer Definition**: In the example, we define two layers. The first is a hidden layer with 2 input neurons and 4 output neurons, using the ReLU activation function. The second is the output layer with 4 input neurons and a single output neuron, using the Sigmoid activation function.
-   **Training**: The neural network is trained using the XOR dataset. We specify the learning rate (0.1) and the number of epochs (10,000). The Mean Squared Error (MSE) loss function is used to calculate the error during backpropagation.
-   **Prediction**: After training, the network is tested on the XOR inputs, and the predicted outputs are printed for each input.
    This example shows how Scorch can be used for simple machine learning tasks like solving the XOR problem. The modular design of Scorch allows you to easily extend it to more complex networks or customize components.

## Customization

Scorch is designed to be flexible and modular. You can add custom activation functions, loss functions, or layer types to suit your specific needs. For example:

-   Add your own activation function.
-   Implement other loss functions.
-   Modify the training process to use advanced optimization techniques like Adam.

This feature makes Scorch an ideal choice for experimenting with different network architectures and algorithms in a simple and understandable way.

## Performance

While Scorch is intended for educational and experimental purposes, its design should be performant enough for small to medium-sized datasets. For larger datasets or more advanced use cases, consider using optimized libraries such as ndarray or tch-rs for higher performance.

## Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, feel free to open an issue or submit a pull request.

### How to contribute:

-   Fork the repository.
-   Create a new branch for your feature or bug fix.
-   Implement your changes and add tests.
-   Run `cargo test` to ensure everything is working.
-   Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
