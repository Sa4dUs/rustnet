use std::env;
use std::error::Error;
use std::fs::File;
use std::ops::Range;
use csv::ReaderBuilder;
use ndarray::Array2;

pub fn read_csv_to_neural_input(file_path: &str, data_indexes: &Range<usize>, result_indexes: &Range<usize>) -> Result<Vec<Vec<Array2<f64>>>, Box<dyn Error>> {
    let mut path = env::current_dir().expect("Failed to get current directory");
    path = path.join("data").join(file_path);

    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut headers: Vec<String> = Vec::new();
    let mut values: Vec<Array2<f64>> = Vec::new();
    let mut results: Vec<Array2<f64>> = Vec::new();

    if let Some(result) = rdr.headers().ok() {
        headers = result.iter().map(|s| s.to_string()).collect();
    }

    let mut numeric_values_set: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let mut numeric_values_data: Vec<f64> = Vec::new();
        let mut numeric_values_results: Vec<f64> = Vec::new();
        let record = result?;

        for (i, value) in record.iter().enumerate() {
            if let Ok(num) = value.parse::<f64>() {
                if data_indexes.contains(&i) {
                    numeric_values_data.push(num);
                } else if result_indexes.contains(&i) {
                    numeric_values_results.push(num);
                }
            }
        }

        for (k, &value) in numeric_values_data.iter().enumerate() {
            if k < numeric_values_set.len() {
                numeric_values_set[k].push(value);
            } else {
                numeric_values_set.push(vec![value]);
            }
        }

        let result_array = Array2::from_shape_vec((numeric_values_results.len(), 1), numeric_values_results)?;
        results.push(result_array.reversed_axes());
    }

    let mut numeric_values_normalized: Vec<Vec<f64>> = Vec::new();
    for row in numeric_values_set.iter().enumerate() {
        numeric_values_normalized.push(normalize(row.0, row.1));
    }

    for i in 0..numeric_values_normalized[0].len() {
        let mut row_values: Vec<f64> = Vec::new();
        for j in 0..numeric_values_normalized.len() {
            row_values.push(numeric_values_normalized[j][i])
        }

        let values_array = Array2::from_shape_vec((row_values.len(), 1), row_values)?;
        values.push(values_array.reversed_axes());
    }

    Ok(vec![values, results])
}

fn normalize(_size: usize, vector: &Vec<f64>) -> Vec<f64> {
    let length = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    vector.iter().map(|&x| x / length).collect()
}