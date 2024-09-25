use std::fmt::Debug;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::io::Result;

use rand::seq::SliceRandom;
use rustnet::{layer::Layer, re_lu_prime};

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 10;
const LEARNING_RATE: f32 = 0.001f32;
const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 64;
const IMAGE_SIZE: usize = 28;
const TRAIN_SPLIT: f32 = 0.8f32;

const TRAIN_IMG_PATH: &str = "data/train-images.idx3-ubyte";
const TRAIN_LBL_PATH: &str = "data/train-labels.idx1-ubyte";

#[derive(Clone, Copy, Debug)]
struct Network<const IN_SIZE: usize, const HID_SIZE: usize, const OUT_SIZE: usize> {
    hidden: Layer<IN_SIZE, HID_SIZE>,
    output: Layer<HID_SIZE, OUT_SIZE>,
}

impl<const IN_SIZE: usize, const HID_SIZE: usize, const OUT_SIZE: usize>
    Network<IN_SIZE, HID_SIZE, OUT_SIZE>
{
    fn new(
        hidden: Layer<IN_SIZE, HID_SIZE>,
        output: Layer<HID_SIZE, OUT_SIZE>,
    ) -> Network<IN_SIZE, HID_SIZE, OUT_SIZE> {
        Network { hidden, output }
    }

    fn train(&mut self, input: [f32; IN_SIZE], label: usize, alpha: f32) {
        let hidden_output: [f32; HID_SIZE] = self.hidden.forward(input);
        let final_output: [f32; OUT_SIZE] = self.output.forward(hidden_output);

        let mut output_grad: [f32; OUT_SIZE] = [0.0f32; OUT_SIZE];

        (0..OUT_SIZE).for_each(|i| {
            output_grad[i] = final_output[i] - ((i == label) as u8 as f32);
        });

        let mut hidden_grad: [f32; HID_SIZE] =
            self.output.backward(hidden_output, output_grad, alpha);
        re_lu_prime(&mut hidden_grad);

        self.hidden.backward(input, hidden_grad, alpha);
    }

    fn predict(&self, input: [f32; IN_SIZE]) -> usize {
        let hidden_output: [f32; HID_SIZE] = self.hidden.forward(input);
        let final_output: [f32; OUT_SIZE] = self.output.forward(hidden_output);

        let mut max_index: usize = 0;

        (1..OUT_SIZE).for_each(|i| {
            if final_output[i] > final_output[max_index] {
                max_index = i;
            }
        });

        max_index
    }
}

fn read_mnist_images(filename: &str) -> io::Result<(Vec<Vec<u8>>, usize, usize)> {
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 16];
    reader.read_exact(&mut header)?;

    let magic_number = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let num_rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let num_cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

    if magic_number != 2051 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number for image file",
        ));
    }

    let mut images = Vec::with_capacity(num_images);
    let mut image_data = vec![0u8; num_rows * num_cols];

    for _ in 0..num_images {
        reader.read_exact(&mut image_data)?;
        images.push(image_data.clone());
    }

    Ok((images, num_rows, num_cols))
}

fn read_mnist_labels(filename: &str) -> io::Result<Vec<u8>> {
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;

    let magic_number = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;

    if magic_number != 2049 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number for label file",
        ));
    }

    let mut labels = vec![0u8; num_labels];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}

fn main() {
    const N: usize = 1_000_000;

    let _ = std::thread::Builder::new()
        .stack_size(size_of::<f32>() * N)
        .spawn(|| -> Result<()> {
            let (images, rows, cols) = read_mnist_images(TRAIN_IMG_PATH)?;
            let labels = read_mnist_labels(TRAIN_LBL_PATH)?;

            assert_eq!(rows * cols, INPUT_SIZE);

            let mut data: Vec<(Vec<u8>, u8)> = images.into_iter().zip(labels).collect();
            let mut rng = rand::thread_rng();
            data.shuffle(&mut rng);

            let train_size = (data.len() as f32 * TRAIN_SPLIT) as usize;
            let test_size = data.len() - train_size;
            let (train_data, test_data) = data.split_at(train_size);

            let hidden: Layer<INPUT_SIZE, HIDDEN_SIZE> = Layer::new(rustnet::re_lu);
            let output: Layer<HIDDEN_SIZE, OUTPUT_SIZE> = Layer::new(rustnet::softmax);
            let mut nn: Network<INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE> =
                Network::new(hidden, output);

            for epoch in 0..EPOCHS {
                let mut total_loss = 0.0;

                for batch in train_data.chunks(BATCH_SIZE) {
                    for (image, label) in batch {
                        let mut img: [f32; INPUT_SIZE] = [0.0f32; INPUT_SIZE];
                        (0..INPUT_SIZE).for_each(|i| {
                            img[i] = image[i] as f32 / 255.0;
                        });

                        nn.train(img, *label as usize, LEARNING_RATE);

                        let hidden_output: [f32; HIDDEN_SIZE] = nn.hidden.forward(img);
                        let final_output: [f32; OUTPUT_SIZE] = nn.output.forward(hidden_output);

                        total_loss += -(final_output[*label as usize] + 1e-10).ln();
                    }
                }

                let mut correct = 0;
                for (image, label) in test_data {
                    let mut img: [f32; INPUT_SIZE] = [0.0f32; INPUT_SIZE];
                    (0..INPUT_SIZE).for_each(|i| {
                        img[i] = image[i] as f32 / 255.0;
                    });

                    println!(
                        "[TEST]\nACTUAL: {}\nEXPECTED: {}\n",
                        nn.predict(img),
                        *label
                    );

                    if nn.predict(img) == *label as usize {
                        correct += 1;
                    }
                }

                let accuracy = correct as f32 / test_size as f32 * 100.0;
                println!(
                    "Epoch {}, Accuracy: {:.2}%, Avg Loss: {:.4}",
                    epoch + 1,
                    accuracy,
                    total_loss / train_size as f32
                );
            }
            Ok(())
        })
        .unwrap()
        .join()
        .unwrap();
}
