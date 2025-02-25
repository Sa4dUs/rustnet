pub struct LossFunction {
    pub function: fn(&[f32], &[f32]) -> f32,
    pub derivative: fn(&[f32], &[f32]) -> Vec<f32>,
}

pub const MSE: LossFunction = LossFunction {
    function: mean_squared_error,
    derivative: mse_derivative,
};

fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (pred - target).powi(2))
        .sum::<f32>()
        / predictions.len() as f32
}

fn mse_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| 2.0 * (pred - target))
        .collect()
}
