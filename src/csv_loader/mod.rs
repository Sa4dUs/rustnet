use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use crate::matrix::MatrixF32;

pub fn read_csv_to_neural_input(file_path: &str, data_indexes: Vec<i32>, result_indexes: Vec<i32>) -> Result<Vec<Vec<MatrixF32>>, Box<dyn Error>> {
    // Abre el archivo
    let file = File::open(file_path)?;

    // Crea un lector CSV con el archivo
    let mut rdr = ReaderBuilder::new().from_reader(file);

    // Vector para almacenar los encabezados
    let mut headers: Vec<String> = Vec::new();
    // Vector para almacenar las filas
    let mut values : Vec<MatrixF32> = Vec::new();
    let mut results : Vec<MatrixF32> = Vec::new();


    // Lee los encabezados
    if let Some(result) = rdr.headers().ok() {
        headers = result.iter().map(|s| s.to_string()).collect();
    }

    // Itera sobre los registros
    for result in rdr.records()
    {
        let mut numeric_values_data: Vec<f32> = Vec::new();
        let mut numeric_values_results: Vec<f32> = Vec::new();
        let record = result?;

        let mut i = 0;
        for value in record.iter()
        {
            if let Ok(num) = value.parse::<f32>()
            {
                if data_indexes.contains(&i)
                {
                    numeric_values_data.push(num);
                }
                else if result_indexes.contains(&i)
                {
                    numeric_values_results.push(num);
                }
            }
            i += 1;
        }

        values.push(MatrixF32::from_vector(vec![numeric_values_data]).t());
        results.push(MatrixF32::from_vector(vec![numeric_values_results]).t());
    }

    Ok(vec![values, results])
}
