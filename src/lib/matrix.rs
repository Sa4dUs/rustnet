use std::ops;
use rand::Rng;
use crate::lib::math::Function;

#[derive(Clone, Debug)]
pub struct MatrixF32(Vec<Vec<f32>>);

impl MatrixF32 {
    pub fn new(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32(vec![vec![0.0; cols]; rows])
    }

    pub fn from_vector(data: Vec<Vec<f32>>) -> MatrixF32 {
        MatrixF32(data)
    }

    pub fn zeros_like(matrix: &MatrixF32) -> MatrixF32 {
        let rows = matrix.get_rows();
        let cols = matrix.get_cols();
        MatrixF32::new(rows, cols)
    }

    pub fn randomized(mut self, scale: f32) -> MatrixF32 {
        let mut rng = rand::thread_rng();
        for i in 0..self.0.len() {
            for j in 0..self.0[i].len() {
                self.0[i][j] = rng.gen_range(-scale..scale);
            }
        }
        self
    }

    pub fn get_rows(&self) -> usize {
        self.0.len()
    }

    pub fn get_cols(&self) -> usize {
        self.0[0].len()
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.0[row][col]
    }

    pub fn t(&self) -> MatrixF32 {
        let rows = self.0.len();
        let cols = self.0[0].len();
        let mut transposed = MatrixF32(vec![vec![0.0; rows]; cols]);
        for i in 0..rows {
            for j in 0..cols {
                transposed.0[j][i] = self.0[i][j];
            }
        }
        transposed
    }

    pub fn elementwise_mul(&self, other: &MatrixF32) -> MatrixF32 {
        let rows = self.get_rows();
        let cols = self.get_cols();

        assert_eq!(rows, other.get_rows(), "Matrix dimensions are not compatible for element-wise multiplication.");
        assert_eq!(cols, other.get_cols(), "Matrix dimensions are not compatible for element-wise multiplication.");

        let mut result = MatrixF32::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.0[i][j] = self.0[i][j] * other.0[i][j];
            }
        }
        result
    }

    pub fn mean_column(&self) -> Self {
        let rows = self.get_rows();
        let cols = self.get_cols();
        let mut result = MatrixF32::new(rows, 1);
        for i in 0..rows {
            let sum: f32 = self.0[i].iter().sum();
            result.0[i][0] = sum / (cols as f32);
        }
        result
    }

    pub fn apply(&self, f: Function) -> Self {
        let rows = self.get_rows();
        let cols = self.get_cols();
        let mut result = MatrixF32::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.0[i][j] = f(self.0[i][j]);
            }
        }
        result
    }
}

impl ops::Add for &MatrixF32 {
    type Output = MatrixF32;

    fn add(self, rhs: &MatrixF32) -> Self::Output {
        let rows = self.get_rows();
        let cols = self.get_cols();

        assert_eq!(rows, rhs.get_rows(), "Matrix dimensions are not compatible for addition.");
        assert_eq!(cols, rhs.get_cols(), "Matrix dimensions are not compatible for addition.");

        let mut result = MatrixF32::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.0[i][j] = self.0[i][j] + rhs.0[i][j];
            }
        }
        result
    }
}

impl ops::Sub for &MatrixF32 {
    type Output = MatrixF32;

    fn sub(self, rhs: &MatrixF32) -> Self::Output {
        let rows = self.get_rows();
        let cols = self.get_cols();

        assert_eq!(rows, rhs.get_rows(), "Matrix dimensions are not compatible for subtraction.");
        assert_eq!(cols, rhs.get_cols(), "Matrix dimensions are not compatible for subtraction.");

        let mut result = MatrixF32::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.0[i][j] = self.0[i][j] - rhs.0[i][j];
            }
        }
        result
    }
}

impl ops::Mul<&MatrixF32> for &MatrixF32 {
    type Output = MatrixF32;

    fn mul(self, rhs: &MatrixF32) -> Self::Output {
        let rows = self.get_rows();
        let cols = rhs.get_cols();
        let common_dim = self.get_cols();

        assert_eq!(common_dim, rhs.get_rows(), "Matrix dimensions are not compatible for multiplication.");

        let mut result = MatrixF32::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..common_dim {
                    result.0[i][j] += self.0[i][k] * rhs.0[k][j];
                }
            }
        }
        result
    }
}

impl ops::Mul<f32> for &MatrixF32 {
    type Output = MatrixF32;

    fn mul(self, scalar: f32) -> Self::Output {
        let rows = self.get_rows();
        let cols = self.get_cols();
        let mut result = MatrixF32::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.0[i][j] = self.0[i][j] * scalar;
            }
        }
        result
    }
}
