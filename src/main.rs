use std::collections::VecDeque;
use ndarray::{Array1, Array2};
use neural_network::lib::csv_loader::read_csv_to_neural_input;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use neural_network::lib::math::{MSE, probability_density_function, RELU};
use neural_network::neural::network::NeuralNetwork;

fn main() {
    let inputs = 0..3;
    let outputs = 3..4;

    let csv_out = read_csv_to_neural_input("train.csv", &inputs, &outputs).expect("csv reading failed");
    let x_train = csv_out[0].clone();
    let y_train = csv_out[1].clone();

    let csv_out = read_csv_to_neural_input("test.csv", &inputs, &outputs).expect("csv reading failed");
    let x_test: Vec<Array2<f64>> = csv_out[0].clone();
    let y_test: Vec<Array2<f64>> = csv_out[1].clone();

    let mut rng = StdRng::seed_from_u64(1);

    let mut nn = NeuralNetwork::new(inputs.len(), vec![(4, RELU), (1, RELU)], &mut rng);

    let loss_f = MSE;
    let learning_rate = 0.05;

    let x: Array2<f64> = x_train[0].clone();
    let y = nn.forward(&x);
    println!("BEFORE {}", y);

    for i in 1..1000 {
        nn.train(&x_train, &y_train, learning_rate, loss_f);
    }

    let y = nn.forward(&x);
    println!("AFTER {}", y);
}
