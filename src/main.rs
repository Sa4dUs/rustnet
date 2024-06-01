use rand::Rng;
use crate::math::{MSE, RELU, SIGMOID};
use crate::matrix::MatrixF32;
use crate::neural_network::NeuralNetwork;

mod matrix;
mod neural_layer;
mod math;
mod neural_network;

fn main() {
    let mut neural_network = NeuralNetwork::new(&vec![(2, SIGMOID), (1, RELU)]);


    let mut x_train = vec![];
    let mut y_train = vec![];

    let sample_size = 100;

    for _ in 1..sample_size {
        let mut rng = rand::thread_rng();

        let a = rng.gen_range(0.0..1.0);
        let b = rng.gen_range(0.0..1.0-a);

        x_train.push(MatrixF32::from_vector(vec![
            vec![a],
            vec![b],
        ]));

        y_train.push(MatrixF32::from_vector(vec![
            vec![a + b],
        ]))
    }

    let x_test = MatrixF32::from_vector(vec![
        vec![10980.452214],
        vec![79091.51290],
    ]);

    let learning_rate = 0.5;

    println!("Initial network state:");
    let initial_output = neural_network.forward(&x_test);
    println!("Output before training: {:?}", initial_output);

    neural_network.train(x_train, y_train, MSE, learning_rate);

    let final_output = neural_network.forward(&x_test);
    println!("Output after training: {:?}", final_output)
}
