use rustnet::{layer::Layer, re_lu_prime};

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 10;
const LEARNING_RATE: f32 = 0.001f32;
const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 64;
const IMAGE_SIZE: usize = 28;
const TRAIN_SPLIT: f32 = 0.8f32;

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

    fn predict(self, input: [f32; IN_SIZE]) -> usize {
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

fn main() {
    let hidden: Layer<INPUT_SIZE, HIDDEN_SIZE> = Layer::new(rustnet::re_lu);
    let output: Layer<HIDDEN_SIZE, OUTPUT_SIZE> = Layer::new(rustnet::softmax);
    let nn: Network<INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE> = Network::new(hidden, output);
}
