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

    let mut neural_network = NeuralNetwork::new(&vec![(inputs.len(), SIGMOID), (6, SIGMOID), (6, SIGMOID),(outputs.len(), SIGMOID)]);


    let csv_out = read_csv_to_neural_input("C:/Users/jorge/Rust/neural-network/mushroom_cleaned.csv", inputs, outputs).expect("TODO: panic message");

    let mut x_train = vec![];
    let mut y_train = vec![];

    x_train = csv_out[0].clone();
    y_train = csv_out[1].clone();
    /*
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
    */
    let learning_rate = 0.5;

    println!("Initial network state:");
    let initial_output = neural_network.forward(&x_train[1035]);
    println!("Output before training: {:?}", initial_output);

    neural_network.train(x_train.clone(), y_train.clone(), MSE, learning_rate);

    let final_output = neural_network.forward(&x_train[1035]);
    println!("Output after training: {:?}", final_output);
    println!("Expected output: {:?}", &y_train[1035]);
}
