use mnist::*;
use ndarray::prelude::*;
use rustnet::layer::Layer;

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 10;
const LEARNING_RATE: f32 = 0.001f32;
const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 64;
const _IMAGE_SIZE: usize = 28;
const _TRAIN_SPLIT: f32 = 0.8f32;

struct Network {
    hidden: Layer,
    output: Layer,
}

fn softmax(input: &mut [f32]) {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;

    for val in input.iter_mut() {
        *val = f32::exp(*val - max);
        sum += *val;
    }

    for val in input.iter_mut() {
        *val /= sum;
    }
}

fn relu(input: &mut [f32]) {
    for val in input.iter_mut() {
        *val = if *val > 0.0 { *val } else { 0.0 };
    }
}

fn train(net: &mut Network, input: &[f32], label: u8, lr: f32) {
    let mut hidden_output = vec![0.0; HIDDEN_SIZE];
    let mut final_output = vec![0.0; OUTPUT_SIZE];

    net.hidden.forward(input, &mut hidden_output);
    relu(&mut hidden_output);

    net.output.forward(&hidden_output, &mut final_output);
    softmax(&mut final_output);

    let mut output_grad = vec![0.0; OUTPUT_SIZE];
    let mut hidden_grad = vec![0.0; HIDDEN_SIZE];

    for i in 0..OUTPUT_SIZE {
        output_grad[i] = final_output[i] - if i == label as usize { 1.0 } else { 0.0 };
    }

    net.output
        .backward(&hidden_output, &output_grad, Some(&mut hidden_grad), lr);

    for i in 0..HIDDEN_SIZE {
        hidden_grad[i] *= if hidden_output[i] > 0.0 { 1.0 } else { 0.0 };
    }

    net.hidden.backward(input, &hidden_grad, None, lr);
}

fn predict(net: &Network, input: &[f32]) -> usize {
    let mut hidden_output = vec![0.0; HIDDEN_SIZE];
    let mut final_output = vec![0.0; OUTPUT_SIZE];

    net.hidden.forward(input, &mut hidden_output);
    relu(&mut hidden_output);

    net.output.forward(&hidden_output, &mut final_output);
    softmax(&mut final_output);

    final_output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn main() {
    let mut net = Network {
        hidden: Layer::new(INPUT_SIZE, HIDDEN_SIZE),
        output: Layer::new(HIDDEN_SIZE, OUTPUT_SIZE),
    };

    let learning_rate = LEARNING_RATE;

    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 255.0);

    let train_labels: Vec<u8> = trn_lbl;
    let train_size = train_labels.len();

    for epoch in 0..EPOCHS {
        let mut total_loss: f32 = 0.0;
        for i in (0..train_size).step_by(BATCH_SIZE) {
            for j in 0..BATCH_SIZE {
                if i + j >= train_size {
                    break;
                }

                let idx = i + j;
                let image = train_data
                    .slice(s![idx, .., ..])
                    .iter()
                    .cloned()
                    .collect::<Vec<f32>>();
                let label = train_labels[idx];

                train(&mut net, &image, label, learning_rate);

                let mut hidden_output = vec![0.0; HIDDEN_SIZE];
                let mut final_output = vec![0.0; OUTPUT_SIZE];
                net.hidden.forward(&image, &mut hidden_output);
                net.output.forward(&hidden_output, &mut final_output);
                softmax(&mut final_output);

                total_loss += -f32::ln(final_output[label as usize] + 1e-10);
            }
        }

        let mut correct: i32 = 0;
        (0..train_size).for_each(|i| {
            let image = train_data
                .slice(s![i, .., ..])
                .iter()
                .cloned()
                .collect::<Vec<f32>>();
            let prediction = predict(&net, &image);

            if prediction == train_labels[i] as usize {
                correct += 1;
            }
        });

        println!(
            "Epoch {}, Accuracy: {}%, Avg Loss: {}",
            epoch + 1,
            correct as f32 / train_size as f32 * 100.0,
            total_loss / train_size as f32
        );
    }
}
