/// `LossFunction` represents a loss function used to evaluate how well the model's predictions
/// match the actual targets during training. It contains two function pointers:
/// one for the loss function itself and another for computing its derivative (gradient),
/// which is used during backpropagation.
///
/// # Fields
/// * `function` - A function that computes the value of the loss function given the model's predictions and the true target values.
/// * `derivative` - A function that computes the gradient of the loss with respect to the predictions, which is used for backpropagation.
///
/// # Example
/// ```
/// let loss = LossFunction {
///     function: mean_squared_error,
///     derivative: mse_derivative,
/// };
/// ```
pub struct LossFunction {
    /// The function that computes the loss value given predictions and targets.
    pub function: fn(&[f32], &[f32]) -> f32,

    /// The function that computes the derivative (gradient) of the loss with respect to the predictions.
    pub derivative: fn(&[f32], &[f32]) -> Vec<f32>,
}

/// `MSE` is a constant that holds the `LossFunction` for Mean Squared Error.
/// It contains both the `mean_squared_error` function and its derivative `mse_derivative`.
/// Mean Squared Error is a commonly used loss function for regression tasks.
///
/// # Example
/// ```
/// let predictions = vec![1.0, 2.0, 3.0];
/// let targets = vec![1.1, 2.1, 3.1];
/// let loss_value = MSE.function(&predictions, &targets);
/// let gradients = MSE.derivative(&predictions, &targets);
/// ```
pub const MSE: LossFunction = LossFunction {
    function: mean_squared_error,
    derivative: mse_derivative,
};

/// `mean_squared_error` computes the Mean Squared Error (MSE) between the predictions and the targets.
/// The MSE is the average of the squared differences between predicted and actual values.
///
/// # Arguments
/// * `predictions` - A slice of `f32` values representing the predicted values from the model.
/// * `targets` - A slice of `f32` values representing the true target values.
///
/// # Returns
/// The Mean Squared Error value, which is a non-negative `f32` representing the average squared difference
/// between the predictions and targets.
///
/// # Example
/// ```
/// let predictions = vec![1.0, 2.0, 3.0];
/// let targets = vec![1.1, 2.1, 3.1];
/// let loss = mean_squared_error(&predictions, &targets);
/// assert_eq!(loss, 0.01);
/// ```
fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (pred - target).powi(2)) // Calculate squared difference
        .sum::<f32>()
        / predictions.len() as f32 // Average the squared differences
}

/// `mse_derivative` computes the derivative of the Mean Squared Error (MSE) loss with respect to the predictions.
/// This derivative is used during backpropagation to update the model's parameters.
///
/// # Arguments
/// * `predictions` - A slice of `f32` values representing the predicted values from the model.
/// * `targets` - A slice of `f32` values representing the true target values.
///
/// # Returns
/// A `Vec<f32>` where each element is the gradient of the loss with respect to the corresponding prediction.
///
/// # Example
/// ```
/// let predictions = vec![1.0, 2.0, 3.0];
/// let targets = vec![1.1, 2.1, 3.1];
/// let gradients = mse_derivative(&predictions, &targets);
/// assert_eq!(gradients, vec![-0.2, -0.2, -0.2]);
/// ```
fn mse_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| 2.0 * (pred - target)) // Derivative of squared error
        .collect()
}
