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

    x_train = csv_out[0].clone();
    y_train = csv_out[1].clone();

    let mut x_test = x_train.clone();
    let mut y_test = y_train.clone();

    let learning_rate = 0.03;

    neural_network.train(x_train.clone(), y_train.clone(), MSE, learning_rate);

    neural_network.test(x_test, y_test, MSE, 0.1);
}
