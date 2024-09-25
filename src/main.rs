use mnist::*;
use ndarray::prelude::*;
use rand::Rng;
use std::alloc::{alloc, Layout};
use std::ptr;

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 10;
const LEARNING_RATE: f32 = 0.001f32;
const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 64;
const _IMAGE_SIZE: usize = 28;
const _TRAIN_SPLIT: f32 = 0.8f32;

struct Layer {
    weights: *mut f32,
    biases: *mut f32,
    input_size: usize,
    output_size: usize,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            weights: ptr::null_mut(),
            biases: ptr::null_mut(),
            input_size: Default::default(),
            output_size: Default::default(),
        }
    }
}

#[derive(Default)]
struct Network {
    hidden: Layer,
    output: Layer,
}

fn softmax(input: *mut f32, size: usize) {
    let mut max: f32 = unsafe { *input.add(0) };
    let mut sum: f32 = 0.0f32;

    for i in 1..size {
        if unsafe { *input.add(i) } > max {
            max = unsafe { *input.add(i) }
        }
    }

    for i in 0..size {
        unsafe { *input.add(i) = f32::exp(*input.add(i) - max) };
        sum += unsafe { *input.add(i) };
    }

    for i in 0..size {
        unsafe { *input.add(i) /= sum };
    }
}

fn init_layer(layer: &mut Layer, in_size: usize, out_size: usize) {
    let n: usize = in_size * out_size;
    let scale: f32 = f32::sqrt(2.0f32 / in_size as f32);

    layer.input_size = in_size;
    layer.output_size = out_size;

    {
        let layout: Layout = Layout::array::<f32>(n).unwrap();
        layer.weights = unsafe { alloc(layout) } as *mut f32;
    }

    {
        let layout: Layout = Layout::array::<f32>(out_size).unwrap();
        layer.biases = unsafe { alloc(layout) } as *mut f32;
        unsafe { ptr::write_bytes(layer.biases, 0, out_size) };
    }

    let mut rng = rand::thread_rng();
    for i in 0..n {
        unsafe {
            *layer.weights.add(i) = ((rng.gen_range(0.0f32..=1.0) - 0.5f32) * 2.0f32) * scale
        };
    }
}

fn forward(layer: &Layer, input: *const f32, output: *mut f32) {
    for i in 0..layer.output_size {
        unsafe { *output.add(i) = *layer.biases.add(i) };
        for j in 0..layer.input_size {
            unsafe {
                *output.add(i) += *input.add(j) * *layer.weights.add(j * layer.output_size + i)
            }
        }
    }
}

fn backward(
    layer: &mut Layer,
    input: *const f32,
    output_grad: *const f32,
    input_grad: *mut f32,
    lr: f32,
) {
    for i in 0..layer.output_size {
        for j in 0..layer.input_size {
            let idx: usize = j * layer.output_size + i;
            let grad: f32 = unsafe { *output_grad.add(i) } * unsafe { *input.add(j) };
            unsafe { *layer.weights.add(idx) -= lr * grad };

            if !input_grad.is_null() {
                unsafe { *input_grad.add(j) += *output_grad.add(i) * *layer.weights.add(idx) };
            }
        }

        unsafe { *layer.biases.add(i) -= lr * *output_grad.add(i) };
    }
}

fn train(net: &mut Network, input: *const f32, label: u8, lr: f32) {
    let mut hidden_output: [f32; HIDDEN_SIZE] = [0.0f32; HIDDEN_SIZE];
    let mut final_output: [f32; OUTPUT_SIZE] = [0.0f32; OUTPUT_SIZE];

    let mut output_grad: [f32; OUTPUT_SIZE] = [0.0f32; OUTPUT_SIZE];
    let mut hidden_grad: [f32; HIDDEN_SIZE] = [0.0f32; HIDDEN_SIZE];

    forward(&net.hidden, input, hidden_output.as_mut_ptr());

    (0..HIDDEN_SIZE).for_each(|i| {
        hidden_output[i] = if hidden_output[i] > 0.0f32 {
            hidden_output[i]
        } else {
            0.0f32
        };
    });

    forward(
        &net.output,
        hidden_output.as_ptr(),
        final_output.as_mut_ptr(),
    );
    softmax(final_output.as_mut_ptr(), OUTPUT_SIZE);

    for i in 0..OUTPUT_SIZE {
        output_grad[i] = final_output[i] - ((i == label as usize) as i32 as f32);
    }
    backward(
        &mut net.output,
        hidden_output.as_ptr(),
        output_grad.as_ptr(),
        hidden_grad.as_mut_ptr(),
        lr,
    );

    for i in 0..HIDDEN_SIZE {
        hidden_grad[i] *= if hidden_output[i] > 0.0f32 {
            1.0f32
        } else {
            0.0f32
        };
    }

    backward(
        &mut net.hidden,
        input,
        hidden_grad.as_ptr(),
        ptr::null_mut(),
        lr,
    );
}

fn predict(net: &Network, input: *const f32) -> usize {
    let mut hidden_output: [f32; HIDDEN_SIZE] = [0.0f32; HIDDEN_SIZE];
    let mut final_output: [f32; OUTPUT_SIZE] = [0.0f32; OUTPUT_SIZE];

    forward(&net.hidden, input, hidden_output.as_mut_ptr());
    (0..HIDDEN_SIZE).for_each(|i| {
        hidden_output[i] = if hidden_output[i] > 0.0f32 {
            hidden_output[i]
        } else {
            0.0f32
        }
    });

    forward(
        &net.output,
        hidden_output.as_ptr(),
        final_output.as_mut_ptr(),
    );
    softmax(final_output.as_mut_ptr(), OUTPUT_SIZE);

    let mut max_index: usize = 0usize;
    for i in 1..OUTPUT_SIZE {
        if final_output[i] > final_output[max_index] {
            max_index = i;
        }
    }

    max_index
}

fn main() {
    let mut net: Network = Network::default();
    init_layer(&mut net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&mut net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    let learning_rate: f32 = LEARNING_RATE;

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

                train(&mut net, image.as_ptr(), label, learning_rate);
                let mut hidden_output: [f32; HIDDEN_SIZE] = [0.0f32; HIDDEN_SIZE];
                let mut final_output: [f32; OUTPUT_SIZE] = [0.0f32; OUTPUT_SIZE];
                forward(&net.hidden, image.as_ptr(), hidden_output.as_mut_ptr());
                forward(
                    &net.output,
                    hidden_output.as_ptr(),
                    final_output.as_mut_ptr(),
                );
                softmax(final_output.as_mut_ptr(), OUTPUT_SIZE);

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
            let prediction = predict(&net, image.as_ptr());

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
