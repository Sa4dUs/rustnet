use std::collections::VecDeque;
use std::fs::soft_link;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::num_traits::signum;
use neural_network::lib::csv_loader::read_csv_to_neural_input;

use neural_network::lib::safe_f64::SafeF64;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use neural_network::lib::math::{get_activation_function, probability_density_function};
use neural_network::lib::math::ActivationFunctionsEnum::{RELU, SIGMOID, SOFTMAX};
use neural_network::lib::math::ErrorFunctionsEnum::{MSE, SOFTMAX_CROSS_ENTROPY};
use neural_network::lib::network_parser::{load_from, save_to};
use neural_network::neural::network::NeuralNetwork;

fn main() {
    let inputs = 0..2;
    let outputs = 2..3;
    let output_values: usize = 2;
    let is_classification: bool = true;
    let training_file_path: &str = "train.csv";
    let test_file_path: &str = "train.csv";

    let csv_out = read_csv_to_neural_input(training_file_path, &inputs, &outputs, is_classification, output_values).expect("csv reading failed");
    let x_train = csv_out[0].clone();
    let y_train = csv_out[1].clone();

    let csv_out = read_csv_to_neural_input(test_file_path, &inputs, &outputs, is_classification, output_values).expect("csv reading failed");
    let x_test: Vec<Array2<SafeF64>> = csv_out[0].clone();
    let y_test: Vec<Array2<SafeF64>> = csv_out[1].clone();

    let mut rng = StdRng::seed_from_u64(1);

    let mut nn = NeuralNetwork::new(inputs.len(), vec![(2, SIGMOID),(output_values, SOFTMAX)], &mut rng);
    //let mut nn = NeuralNetwork::new_empty();
    //nn.load("Testo", &mut rng);

    let loss_f = SOFTMAX_CROSS_ENTROPY;
    let learning_rate = SafeF64::new(0.03);
    let epochs = 1000;
    let threshold = SafeF64::new(0.05);

    nn.train(&x_train, &y_train, learning_rate, loss_f, epochs);
    nn.test(&x_test, &y_test, loss_f, threshold);

    nn.save("After");

}
