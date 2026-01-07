#[cfg(test)]
mod tests {
    use crate::activation::{relu, relu_derivative, sigmoid, sigmoid_derivative};
    use crate::matrix::Matrix;
    use crate::network::NeuralNetwork;
    use crate::data::SyntheticDataGenerator;

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
    
    #[test]
    fn test_model_export_and_reload() {
        use std::fs;
        use std::env;
        
        // Train a simple network
        let mut data_gen = SyntheticDataGenerator::new(42);
        let (train_inputs, train_targets) = data_gen.generate_linear_separable(50, 2);
        
        let mut network = NeuralNetwork::new(vec![2, 4, 1], 0.1, 12345);
        network.train(&train_inputs, &train_targets, 100);
        
        // Export to JSON in temp directory
        let test_path = env::temp_dir().join("test_model_export.json");
        network.save_to_json(test_path.to_str().unwrap()).expect("Failed to save model");
        
        // Verify file exists
        assert!(fs::metadata(&test_path).is_ok(), "Model file should exist");
        
        // Load the model back
        let loaded_network = NeuralNetwork::load_from_json(test_path.to_str().unwrap())
            .expect("Failed to load model");
        
        // Verify architectures match
        assert_eq!(network.layer_sizes, loaded_network.layer_sizes);
        assert_eq!(network.weights.len(), loaded_network.weights.len());
        assert_eq!(network.biases.len(), loaded_network.biases.len());
        
        // Verify predictions match
        for input in train_inputs.iter().take(5) {
            let original_pred = network.predict(input);
            let loaded_pred = loaded_network.predict(input);
            
            for i in 0..original_pred.data.len() {
                let diff = (original_pred.data[i] - loaded_pred.data[i]).abs();
                assert!(diff < 1e-10, "Predictions should match exactly");
            }
        }
        
        // Clean up
        fs::remove_file(test_path).ok();
    }
    
    #[test]
    fn test_json_serialization_format() {
        // Create a simple network
        let network = NeuralNetwork::new(vec![2, 3, 1], 0.1, 42);
        
        // Export to JSON string
        let json_str = network.to_json_str().expect("Failed to serialize");
        
        // Verify JSON contains expected fields
        assert!(json_str.contains("layer_sizes"));
        assert!(json_str.contains("learning_rate"));
        assert!(json_str.contains("weights"));
        assert!(json_str.contains("biases"));
        
        // Load from JSON string
        let loaded_network = NeuralNetwork::from_json_str(&json_str)
            .expect("Failed to deserialize");
        
        // Verify structure
        assert_eq!(network.layer_sizes, loaded_network.layer_sizes);
        assert_eq!(network.learning_rate, loaded_network.learning_rate);
    }
}
