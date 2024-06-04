use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use ndarray::Array2;
use serde_json;

#[derive(Serialize, Deserialize)]
pub struct Data {
    array: Vec<Array2<f64>>,
    label: String,
}

pub fn save_to(data: Vec<(Vec<Array2<f64>>, String)>, dir: &str) -> std::io::Result<()> {
    let path = get_dir(dir);
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    let data: Vec<Data> = data.into_iter().map(|(array, label)| Data { array, label }).collect();

    serde_json::to_writer(writer, &data)?;
    Ok(())
}

pub fn load_from(dir: &str) -> std::io::Result<Vec<(Vec<Array2<f64>>, String)>> {
    let path = get_dir(dir);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let data: Vec<Data> = serde_json::from_reader(reader)?;

    let data: Vec<(Vec<Array2<f64>>, String)> = data.into_iter().map(|Data { array, label }| (array, label)).collect();

    Ok(data)
}

fn get_dir(file_dir: &str) -> PathBuf {
    let mut path = env::current_dir().expect("Failed to get current directory");
    path = path.join("saves").join(file_dir);
    path
}
