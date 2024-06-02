use std::env;
use std::error::Error;
use std::fs::File;
use std::ops::Range;
use csv::ReaderBuilder;
use crate::matrix::MatrixF32;

// Función para normalizar un vector de valores entre 0 y 1
fn normalize(values: &Vec<f32>) -> Vec<f32> {
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    values.iter().map(|&x| (x - min) / (max - min)).collect()
}

pub fn read_csv_to_neural_input(file_path: &str, data_indexes: &Range<i32>, result_indexes: &Range<i32>) -> Result<Vec<Vec<MatrixF32>>, Box<dyn Error>> {
    // Path relativo
    let mut path = env::current_dir().expect("Failed to get current directory");
    // Agregar la ruta relativa al archivo CSV
    path = path.join("data").join(file_path);

    // Abre el archivo
    let file = File::open(path)?;

    // Crea un lector CSV con el archivo
    let mut rdr = ReaderBuilder::new().from_reader(file);

    // Vector para almacenar los encabezados
    let mut headers: Vec<String> = Vec::new();
    // Vector para almacenar las filas
    let mut values: Vec<MatrixF32> = Vec::new();
    let mut results: Vec<MatrixF32> = Vec::new();

    // Lee los encabezados
    if let Some(result) = rdr.headers().ok() {
        headers = result.iter().map(|s| s.to_string()).collect();
    }

    // Vectores para almacenar temporalmente los datos antes de normalizarlos
    let mut all_data: Vec<Vec<f32>> = Vec::new();

    // Itera sobre los registros
    for result in rdr.records() {
        let mut numeric_values_data: Vec<f32> = Vec::new();
        let mut numeric_values_results: Vec<f32> = Vec::new();
        let record = result?;

        let mut i = 0;
        for value in record.iter() {
            if let Ok(num) = value.parse::<f32>() {
                if data_indexes.contains(&i) {
                    numeric_values_data.push(num);
                } else if result_indexes.contains(&i) {
                    numeric_values_results.push(num);
                }
            }
            i += 1;
        }

        all_data.push(numeric_values_data);
        results.push(MatrixF32::from_vector(vec![numeric_values_results]).t());
    }

    // Normalizar los valores
    for data in all_data {
        let normalized_data = normalize(&data);
        values.push(MatrixF32::from_vector(vec![normalized_data]).t());
    }

    Ok(vec![values, results])
}
