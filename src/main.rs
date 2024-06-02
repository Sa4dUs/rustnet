use rand::Rng;
use crate::csv_loader::read_csv_to_neural_input;
use crate::math::{MSE, RELU, SIGMOID};
use crate::matrix::MatrixF32;
use crate::neural_network::NeuralNetwork;

mod matrix;
mod neural_layer;
mod math;
mod neural_network;
mod csv_loader;

fn main() {
    let inputs = vec![0,1,2,3,4,5,6,7];
    let outputs = vec![8];

    let mut neural_network = NeuralNetwork::new(&vec![(inputs.len(), SIGMOID), (8, SIGMOID), (4, SIGMOID),(outputs.len(), SIGMOID)]);

    let csv_out = read_csv_to_neural_input("mushroom_cleaned.csv", inputs, outputs).expect("csv reading failed");

    let mut x_train = vec![];
    let mut y_train = vec![];

    let x_test = 1035;

    x_train = csv_out[0].clone();
    y_train = csv_out[1].clone();

    let learning_rate = 0.03;

    println!("Initial network state:");
    let initial_output = neural_network.forward(&x_train[x_test]);
    println!("Output before training: {:?}", initial_output);

    neural_network.train(x_train.clone(), y_train.clone(), MSE, learning_rate);

    let final_output = neural_network.forward(&x_train[x_test]);
    println!("Output after training: {:?}", final_output);
    println!("Expected output: {:?}", &y_train[x_test]);
}
