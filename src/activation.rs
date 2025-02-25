/// `ActivationFunction` is a type alias for a function that takes a slice of `f32` values
/// (the input to the activation function) and returns a `Vec<f32>` (the output of the activation function).
/// This type alias is used to define various activation functions in the neural network.
pub type ActivationFunction = fn(&[f32]) -> Vec<f32>;

/// `softmax` is an activation function that converts a vector of raw input values (often called logits)
/// into a probability distribution, where the sum of the output values equals 1.0. It is commonly used
/// in the output layer of neural networks for classification tasks.
///
/// # Arguments
/// * `input` - A slice of `f32` values representing the input to the softmax function (often logits).
///
/// # Returns
/// * A `Vec<f32>` where each element is a probability corresponding to each input value.
///   The probabilities are scaled such that their sum equals 1.0.
///
/// # Example
/// ```
/// let input = vec![1.0, 2.0, 3.0];
/// let output = softmax(&input);
/// assert_eq!(output.len(), input.len());
/// assert!(output.iter().sum::<f32>() - 1.0 < 1e-6);  // The sum of probabilities should be 1.0.
/// ```
pub fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = input.iter().map(|&x| f32::exp(x - max)).collect();
    let sum: f32 = exp_values.iter().sum();

    exp_values.iter().map(|&x| x / sum).collect()
}

/// `relu` is the Rectified Linear Unit (ReLU) activation function, which outputs the input value
/// if it is positive and zero otherwise. ReLU is one of the most commonly used activation functions
/// in neural networks, especially for hidden layers.
///
/// # Arguments
/// * `input` - A slice of `f32` values representing the input to the ReLU function.
///
/// # Returns
/// * A `Vec<f32>` where each element is either the input value (if it is greater than 0) or 0 (if the input is negative).
///
/// # Example
/// ```
/// let input = vec![-1.0, 0.0, 1.0];
/// let output = relu(&input);
/// assert_eq!(output, vec![0.0, 0.0, 1.0]);
/// ```
pub fn relu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect()
}

/// `linear` is a linear activation function that simply returns the input values as they are.
/// This function is commonly used in the output layer of a neural network, especially in regression tasks,
/// where no transformation of the output values is needed.
///
/// # Arguments
/// * `input` - A slice of `f32` values representing the input to the linear function.
///
/// # Returns
/// * A `Vec<f32>` that is identical to the input vector, as no transformation is applied.
///
/// # Example
/// ```
/// let input = vec![1.0, 2.0, 3.0];
/// let output = linear(&input);
/// assert_eq!(output, input);
/// ```
pub fn linear(input: &[f32]) -> Vec<f32> {
    input.to_vec()
}
