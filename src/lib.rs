pub mod layer;

pub type ActivationFunction = fn(&mut [f32]);

pub fn re_lu(input: &mut [f32]) {
    (0..input.len()).for_each(|i| {
        input[i] = f32::max(input[i], 0.0f32);
    });
}

pub fn re_lu_prime(input: &mut [f32]) {
    (0..input.len()).for_each(|i| {
        input[i] *= if input[i] > 0.0f32 { 1.0f32 } else { 0.0f32 };
    });
}

pub fn softmax(input: &mut [f32]) {
    let mut max: f32 = 0.0;
    let mut sum: f32 = 0.0;

    (0..input.len()).for_each(|i| max = f32::max(input[i], max));
    (0..input.len()).for_each(|i| {
        input[i] = f32::exp(input[i] - max);
        sum += input[i];
    });
    (0..input.len()).for_each(|i| input[i] /= sum);
}
