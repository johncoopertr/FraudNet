#[cfg(test)]
mod tests {
    use crate::activation::{relu, relu_derivative, sigmoid, sigmoid_derivative};
    use crate::matrix::Matrix;

    #[test]
    fn test_relu() {
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_relu_derivative() {
        assert_eq!(relu_derivative(5.0), 1.0);
        assert_eq!(relu_derivative(-3.0), 0.0);
        assert_eq!(relu_derivative(0.0), 0.0);
    }

    #[test]
    fn test_sigmoid() {
        let result = sigmoid(0.0);
        assert!((result - 0.5).abs() < 1e-10);

        let result = sigmoid(100.0);
        assert!((result - 1.0).abs() < 1e-10);

        let result = sigmoid(-100.0);
        assert!(result < 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative() {
        // At x=0, sigmoid'(0) = 0.25
        let result = sigmoid_derivative(0.0);
        assert!((result - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::zeros(3, 2);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 2);
        assert_eq!(m.data.len(), 6);
    }

    #[test]
    fn test_matrix_get_set() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(1, 0), 3.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_dot() {
        // [[1, 2],    [[5, 6],     [[19, 22],
        //  [3, 4]] *   [7, 8]]  =   [43, 50]]
        let m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1.dot(&m2);

        assert_eq!(result.get(0, 0), 19.0);
        assert_eq!(result.get(0, 1), 22.0);
        assert_eq!(result.get(1, 0), 43.0);
        assert_eq!(result.get(1, 1), 50.0);
    }

    #[test]
    fn test_matrix_add() {
        let m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1.add(&m2);

        assert_eq!(result.get(0, 0), 6.0);
        assert_eq!(result.get(0, 1), 8.0);
        assert_eq!(result.get(1, 0), 10.0);
        assert_eq!(result.get(1, 1), 12.0);
    }

    #[test]
    fn test_matrix_sub() {
        let m1 = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let m2 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = m1.sub(&m2);

        assert_eq!(result.get(0, 0), 4.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 4.0);
        assert_eq!(result.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_hadamard() {
        let m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let m2 = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = m1.hadamard(&m2);

        assert_eq!(result.get(0, 0), 5.0);
        assert_eq!(result.get(0, 1), 12.0);
        assert_eq!(result.get(1, 0), 21.0);
        assert_eq!(result.get(1, 1), 32.0);
    }

    #[test]
    fn test_matrix_scale() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = m.scale(2.0);

        assert_eq!(result.get(0, 0), 2.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 6.0);
        assert_eq!(result.get(1, 1), 8.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = m.transpose();

        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 2);
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(1, 0), 2.0);
        assert_eq!(result.get(1, 1), 5.0);
        assert_eq!(result.get(2, 0), 3.0);
        assert_eq!(result.get(2, 1), 6.0);
    }

    #[test]
    fn test_matrix_sum() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.sum(), 10.0);
    }

    #[test]
    fn test_matrix_mean() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.mean(), 2.5);
    }
}
