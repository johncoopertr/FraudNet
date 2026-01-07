use crate::matrix::Matrix;

/// ReLU activation function: max(0, x)
pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Derivative of ReLU: 1 if x > 0, else 0
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Sigmoid activation function: 1 / (1 + e^(-x))
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Apply ReLU to a matrix
pub fn relu_matrix(m: &Matrix) -> Matrix {
    m.map(relu)
}

/// Apply ReLU derivative to a matrix
pub fn relu_derivative_matrix(m: &Matrix) -> Matrix {
    m.map(relu_derivative)
}

/// Apply Sigmoid to a matrix
pub fn sigmoid_matrix(m: &Matrix) -> Matrix {
    m.map(sigmoid)
}

/// Apply Sigmoid derivative to a matrix
pub fn sigmoid_derivative_matrix(m: &Matrix) -> Matrix {
    m.map(sigmoid_derivative)
}
