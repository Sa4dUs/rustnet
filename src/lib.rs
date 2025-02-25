pub mod activation;
pub mod layer;
pub mod loss;
pub mod neural_network;

use activation::ActivationFunction;

pub struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    pub input_size: usize,
    pub output_size: usize,
    activation_function: ActivationFunction,
}

pub struct NeuralNetwork<'a> {
    input_size: usize,
    output_size: usize,
    layers: &'a mut [Layer],
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_trivial() {
        assert_eq!(1, 1);
    }
}
