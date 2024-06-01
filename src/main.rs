use std::ptr::null;
use crate::math::{MSE, SIGMOID};
use crate::matrix::MatrixF32;
use crate::neural_layer::NeuralLayer;
use crate::neural_network::NeuralNetwork;

mod matrix;
mod neural_layer;
mod math;
mod neural_network;

fn main() {
    let mut neural_network = NeuralNetwork::new(&vec![(2, SIGMOID), (4, SIGMOID), (2, SIGMOID)]);

    let input = MatrixF32::from_vector(vec![
        vec![0.1],
        vec![0.3],
    ]);

    let target_output = MatrixF32::from_vector(vec![
        vec![0.5],
        vec![0.6],
    ]);

    let learning_rate = 0.1;

    let output = neural_network.forward(&input);
    println!("Output before backpropagation: {:?}", output);

    neural_network.backward(input.clone(), target_output, MSE, learning_rate);

    let output_after_backprop = neural_network.forward(&input);
    println!("Output after backpropagation: {:?}", output_after_backprop);
}
