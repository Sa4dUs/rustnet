use std::env;
use std::error::Error;
use std::fs::File;
use std::ops::Range;
use csv::ReaderBuilder;
use ndarray::{Array, Array2, Axis};
use ndarray_rand::rand_distr::num_traits::real::Real;
use crate::lib::safe_f64::SafeF64;

pub fn read_csv_to_neural_input(file_path: &str, data_indexes: &Range<usize>, result_indexes: &Range<usize>, is_classification: bool, classification_values: usize) -> Result<Vec<Vec<Array2<SafeF64>>>, Box<dyn Error>> {
    //Setup file path
    let mut path = env::current_dir().expect("Failed to get current directory");
    path = path.join("data").join(file_path);

    //Initiate file reader
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    //Ready vector outputs
    let mut headers: Vec<String> = Vec::new();
    let mut inputs: Vec<Array2<SafeF64>> = Vec::new();
    let mut outputs: Vec<Array2<SafeF64>> = Vec::new();

    if let Some(result) = rdr.headers().ok()
    {
        headers = result.iter().map(|s| s.to_string()).collect();
    }

    let input_size = data_indexes.len();
    let output_size = classification_values;

    //Iterate though csv rows
    for result in rdr.records()
    {

        let record = result?;

        let mut inputs_temp = vec![SafeF64::new(0.0); input_size];
        let mut outputs_temp: Vec<SafeF64> = vec![SafeF64::new(0.0); output_size];

        //Iterate through columns in row
        let mut j = 0;
        for (i, value) in record.iter().enumerate() {
            if let Ok(num) =value.parse::<f64>()
            {
                let num = SafeF64::new(num);
                if data_indexes.contains(&i)
                {
                    //Push inputs to temp
                    inputs_temp[i - j] = num;
                } else if result_indexes.contains(&i)
                {
                    if is_classification
                    {
                        //Parse from SafeF64 to usize
                        let mut out_index: usize = 0;
                        if num >= SafeF64::new(0.0) && num.0.is_finite()
                        {
                            out_index = num.0.round() as usize;
                        }

                        // Push outputs to temp
                        outputs_temp[out_index] = SafeF64::new(1.0);
                    }
                    else
                    {
                        //Push outputs to temp
                        outputs_temp[j] = num;
                    }
                    j += 1;
                }
            }
        }

        // From Vec to Array2
        let input_array = Array2::from_shape_vec((1, input_size), inputs_temp)?;
        let output_array = Array2::from_shape_vec((1, output_size), outputs_temp)?;

        //Push colum matrix to function outputs
        inputs.push(input_array);
        outputs.push(output_array);

    }

    Ok(vec![inputs, outputs])
}