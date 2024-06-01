use crate::math::{MSE, SIGMOID};
use crate::matrix::MatrixF32;
use crate::neural_network::NeuralNetwork;

mod matrix;
mod neural_layer;
mod math;
mod neural_network;

fn main() {
    let mut neural_network = NeuralNetwork::new(&vec![(2, SIGMOID), (4, SIGMOID), (1, SIGMOID)]);

    let input = MatrixF32::from_vector(vec![
        vec![0.1],
        vec![0.3],
    ]);

    let target_output = MatrixF32::from_vector(vec![
        vec![0.0],
    ]);

    let learning_rate = 0.1;

    println!("Initial network state:");
    let initial_output = neural_network.forward(&input);
    println!("Output before backpropagation: {:?}", initial_output);

    let epochs = 1000;
    for epoch in 0..epochs {
        neural_network.forward(&input);
        neural_network.backward(input.clone(), target_output.clone(), MSE, learning_rate);

        let output = neural_network.forward(&input);
        println!("Epoch {}: Output: {:?}", epoch, output);
    }

    let final_output = neural_network.forward(&input);
    println!("Final output after training: {:?}", final_output);
}
