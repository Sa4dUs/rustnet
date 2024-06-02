use std::env;
use std::error::Error;
use std::fs::File;
use std::ops::Range;
use csv::ReaderBuilder;
use crate::matrix::MatrixF32;

pub fn read_csv_to_neural_input(file_path: &str, data_indexes: &Range<i32>, result_indexes: &Range<i32>) -> Result<Vec<Vec<MatrixF32>>, Box<dyn Error>> {
    //Path relativo
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
    let mut values : Vec<MatrixF32> = Vec::new();
    let mut results : Vec<MatrixF32> = Vec::new();


    // Lee los encabezados
    if let Some(result) = rdr.headers().ok() {
        headers = result.iter().map(|s| s.to_string()).collect();
    }

    let mut numeric_values_set: Vec<Vec<f32>> = Vec::new();
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

        for (k, &value) in numeric_values_data.iter().enumerate()
        {
            if k < numeric_values_set.len() {
                numeric_values_set[k].push(value);
            } else {
                numeric_values_set.push(vec![value]);
            }
        }

        results.push(MatrixF32::from_vector(vec![numeric_values_results]).t());
    }

    let mut numeric_values_normalized: Vec<Vec<f32>> = Vec::new();
    //Normalize
    for row in numeric_values_set.iter().enumerate()
    {
        numeric_values_normalized.push(normalize(row.0, row.1));
    }

    for i in 0..numeric_values_normalized[0].len()
    {
        let mut row_values: Vec<f32> = Vec::new();
        for j in 0..numeric_values_normalized.len()
        {
            row_values.push(numeric_values_normalized[j][i])
        }

        values.push(MatrixF32::from_vector(vec![row_values]).t())
    }

    Ok(vec![values, results])
}

fn normalize(size: usize, vector:  &Vec<f32>) -> Vec<f32>
{
    // Calcular la longitud (norma) del vector TODO Esta wea de normalizar
    let length = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Dividir cada componente del vector por la longitud
    vector.iter().map(|&x| x / length).collect()
}