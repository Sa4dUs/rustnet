use mnist::*;
use ndarray::prelude::*;
use rustnet::{
    activation::{relu, softmax},
    loss::MSE,
    Layer, NeuralNetwork,
};

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 128;
const OUTPUT_SIZE: usize = 10;
const LEARNING_RATE: f32 = 0.005;
const EPOCHS: usize = 50;
const TRAIN_FRACTION: f32 = 0.8;

fn one_hot_encode(labels: &[u8], num_classes: usize) -> Vec<Vec<f32>> {
    labels
        .iter()
        .map(|&label| {
            let mut one_hot = vec![0.0; num_classes];
            one_hot[label as usize] = 1.0;
            one_hot
        })
        .collect()
}

fn argmax(output: &[f32]) -> usize {
    output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

fn main() {
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(1000)
        .validation_set_length(100)
        .test_set_length(100)
        .finalize();

    let data: Vec<Vec<f32>> = Array3::from_shape_vec((1000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 255.0)
        .to_shape((1000, 784))
        .unwrap()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let labels = one_hot_encode(&trn_lbl, OUTPUT_SIZE);

    let train_size = (TRAIN_FRACTION * data.len() as f32) as usize;
    let (train_data, test_data) = data.split_at(train_size);
    let (train_labels, test_labels) = labels.split_at(train_size);

    let mut binding = [
        Layer::new(INPUT_SIZE, HIDDEN_SIZE, relu),
        Layer::new(HIDDEN_SIZE, OUTPUT_SIZE, softmax),
    ];
    let mut nn = NeuralNetwork::new(&mut binding);

    nn.train(train_data, train_labels, LEARNING_RATE, MSE, EPOCHS);

    let mut correct_predictions = 0;
    for (input, label) in test_data.iter().zip(test_labels.iter()) {
        let output = nn.forward(input);
        let predicted = argmax(&output);
        let actual = argmax(label);

        if predicted == actual {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f32 / test_data.len() as f32 * 100.0;
    println!("Model accuracy on test data: {:.2}%", accuracy);
}
