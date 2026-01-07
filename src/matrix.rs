use crate::utils::SimpleRng;

/// A simple matrix implementation using stdlib only
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a new matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create a new matrix from a flat vector
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length must match rows * cols"
        );
        Matrix { rows, cols, data }
    }

    /// Create a new matrix with random values between -1 and 1
    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let data = (0..rows * cols)
            .map(|_| rng.next_f64() * 2.0 - 1.0)
            .collect();
        Matrix { rows, cols, data }
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Matrix multiplication: self * other
    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Cannot multiply matrices: dimensions don't match"
        );

        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }

    /// Element-wise addition
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Apply a function to each element
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Sum all elements in the matrix
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Calculate mean of all elements
    pub fn mean(&self) -> f64 {
        self.sum() / (self.rows * self.cols) as f64
    }
}
