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
        .map(|(pred, target)| (pred - target).powi(2))
        .sum::<f32>()
        / predictions.len() as f32
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
        .map(|(pred, target)| 2.0 * (pred - target))
        .collect()
}

/// `MAE` is a constant that holds the `LossFunction` for Mean Absolute Error.
/// It contains both the `mean_absolute_error` function and its derivative `mae_derivative`.
/// Mean Absolute Error is commonly used in regression tasks where we want to penalize the absolute difference
/// between the predictions and targets.
///
/// # Example
/// ```
/// let predictions = vec![1.0, 2.0, 3.0];
/// let targets = vec![1.1, 2.1, 3.1];
/// let loss_value = MAE.function(&predictions, &targets);
/// let gradients = MAE.derivative(&predictions, &targets);
/// ```
pub const MAE: LossFunction = LossFunction {
    function: mean_absolute_error,
    derivative: mae_derivative,
};

/// `mean_absolute_error` computes the Mean Absolute Error (MAE) between the predictions and the targets.
/// The MAE is the average of the absolute differences between predicted and actual values.
///
/// # Arguments
/// * `predictions` - A slice of `f32` values representing the predicted values from the model.
/// * `targets` - A slice of `f32` values representing the true target values.
///
/// # Returns
/// The Mean Absolute Error value, which is a non-negative `f32` representing the average absolute difference
/// between the predictions and targets.
///
/// # Example
/// ```
/// let predictions = vec![1.0, 2.0, 3.0];
/// let targets = vec![1.1, 2.1, 3.1];
/// let loss = mean_absolute_error(&predictions, &targets);
/// assert_eq!(loss, 0.1);
/// ```
///
fn mean_absolute_error(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (pred - target).abs())
        .sum::<f32>()
        / predictions.len() as f32
}

/// `mae_derivative` computes the derivative of the Mean Absolute Error (MAE) loss with respect to the predictions.
/// The derivative of MAE is 1 or -1 depending on whether the prediction is greater or smaller than the target.
/// This is used during backpropagation to update the model's parameters.
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
/// let gradients = mae_derivative(&predictions, &targets);
/// assert_eq!(gradients, vec![-1.0, -1.0, -1.0]);
/// ```
///
fn mae_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| if pred > target { 1.0 } else { -1.0 })
        .collect()
}

/// `CrossEntropy` is a constant that holds the `LossFunction` for Cross-Entropy loss.
/// It contains both the `cross_entropy` function and its derivative `cross_entropy_derivative`.
/// Cross-Entropy Loss is used primarily in classification tasks.
///
/// # Example
/// ```
/// let predictions = vec![0.1, 0.9];
/// let targets = vec![0.0, 1.0];
/// let loss_value = CrossEntropy.function(&predictions, &targets);
/// let gradients = CrossEntropy.derivative(&predictions, &targets);
/// ```
pub const CROSS_ENTROPY: LossFunction = LossFunction {
    function: cross_entropy,
    derivative: cross_entropy_derivative,
};

/// `cross_entropy` computes the Cross-Entropy loss between the predictions and the targets.
/// Cross-Entropy is commonly used in classification tasks, particularly in binary and multi-class problems.
///
/// # Arguments
/// * `predictions` - A slice of `f32` values representing the predicted probabilities from the model (between 0 and 1).
/// * `targets` - A slice of `f32` values representing the true class labels (usually 0 or 1 for binary classification).
///
/// # Returns
/// The Cross-Entropy loss value, which is a non-negative `f32` representing the difference between the true distribution
/// and the predicted distribution.
///
/// # Example
/// ```
/// let predictions = vec![0.1, 0.9];
/// let targets = vec![0.0, 1.0];
/// let loss = cross_entropy(&predictions, &targets);
/// assert_eq!(loss, 0.105360516);
/// ```
///
fn cross_entropy(predictions: &[f32], targets: &[f32]) -> f32 {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| {
            if *target == 1.0 {
                -target * pred.ln()
            } else {
                -(1.0 - target) * (1.0 - pred).ln()
            }
        })
        .sum::<f32>()
}

/// `cross_entropy_derivative` computes the derivative of the Cross-Entropy loss with respect to the predictions.
/// This derivative is used during backpropagation to update the model's parameters in classification tasks.
///
/// # Arguments
/// * `predictions` - A slice of `f32` values representing the predicted probabilities from the model.
/// * `targets` - A slice of `f32` values representing the true target values (0 or 1).
///
/// # Returns
/// A `Vec<f32>` where each element is the gradient of the loss with respect to the corresponding prediction.
///
/// # Example
/// ```
/// let predictions = vec![0.1, 0.9];
/// let targets = vec![0.0, 1.0];
/// let gradients = cross_entropy_derivative(&predictions, &targets);
/// assert_eq!(gradients, vec![ -1.1111112,  1.1111112 ]);
/// ```
///
fn cross_entropy_derivative(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| {
            if *target == 1.0 {
                -1.0 / pred
            } else {
                1.0 / (1.0 - pred)
            }
        })
        .collect()
}
