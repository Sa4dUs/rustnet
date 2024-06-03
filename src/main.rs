use std::collections::VecDeque;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::num_traits::signum;
use neural_network::lib::csv_loader::read_csv_to_neural_input;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use neural_network::lib::math::{CROSS_ENTROPY, MSE, probability_density_function, RELU, SIGMOID};
use neural_network::neural::network::NeuralNetwork;

fn main() {
    let inputs = 0..8;
    let outputs = 8..9;

    let csv_out = read_csv_to_neural_input("mushroom_cleaned.csv", &inputs, &outputs).expect("csv reading failed");
    let x_train = csv_out[0].clone();
    let y_train = csv_out[1].clone();

    let csv_out = read_csv_to_neural_input("mushroom_cleaned.csv", &inputs, &outputs).expect("csv reading failed");
    let x_test: Vec<Array2<f64>> = csv_out[0].clone();
    let y_test: Vec<Array2<f64>> = csv_out[1].clone();

    let mut rng = StdRng::seed_from_u64(1);

    let mut nn = NeuralNetwork::new(inputs.len(), vec![(6, SIGMOID), (4, SIGMOID), (outputs.len(), SIGMOID)], &mut rng);

    let loss_f = CROSS_ENTROPY;
    let learning_rate = 0.03;
    let epochs = 10;
    let threshold = 0.05;

    nn.train(&x_train, &y_train, learning_rate, loss_f, epochs);
    nn.test(&x_test, &y_test, loss_f, threshold);
}
