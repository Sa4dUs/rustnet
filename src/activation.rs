pub type ActivationFunction = fn(&[f32]) -> Vec<f32>;

pub fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = input.iter().map(|&x| f32::exp(x - max)).collect();
    let sum: f32 = exp_values.iter().sum();

    exp_values.iter().map(|&x| x / sum).collect()
}

pub fn relu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect()
}

pub fn linear(input: &[f32]) -> Vec<f32> {
    input.to_vec()
}
