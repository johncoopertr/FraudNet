#[cfg(test)]
mod tests {
    use crate::activation::{relu, relu_derivative, sigmoid, sigmoid_derivative};
    use crate::matrix::Matrix;
    use crate::network::NeuralNetwork;
    use crate::data::SyntheticDataGenerator;
    use crate::cnn::MnistCNN;

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

    #[test]
    fn test_fraud_data_generation() {
        use crate::data::FraudDataGenerator;
        
        let mut fraud_gen = FraudDataGenerator::new(42);
        let (inputs, targets) = fraud_gen.generate_fraud_data(100, 0.30);
        
        // Verify correct number of samples
        assert_eq!(inputs.len(), 100);
        assert_eq!(targets.len(), 100);
        
        // Verify fraud ratio (30 fraudulent, 70 legitimate)
        let fraud_count: usize = targets.iter()
            .filter(|t| t.get(0, 0) > 0.5)
            .count();
        assert_eq!(fraud_count, 30);
        
        // Verify input dimensions (15 features)
        for input in &inputs {
            assert_eq!(input.rows, 15);
            assert_eq!(input.cols, 1);
        }
        
        // Verify output dimensions (1 target)
        for target in &targets {
            assert_eq!(target.rows, 1);
            assert_eq!(target.cols, 1);
        }
        
        // Verify feature values are in valid range [0, 1]
        for input in &inputs {
            for i in 0..15 {
                let val = input.get(i, 0);
                assert!(val >= 0.0 && val <= 1.0, 
                    "Feature {} value {} out of range [0, 1]", i, val);
            }
        }
    }

    #[test]
    fn test_fraud_detection_network() {
        use crate::data::FraudDataGenerator;
        
        let mut fraud_gen = FraudDataGenerator::new(123);
        
        // Generate small dataset for quick test
        let (train_inputs, train_targets) = fraud_gen.generate_fraud_data(100, 0.30);
        let (test_inputs, test_targets) = fraud_gen.generate_fraud_data(50, 0.30);
        
        // Create fraud detection network (smaller for testing)
        let mut network = NeuralNetwork::new(vec![15, 32, 16, 8, 1], 0.1, 12345);
        
        // Train the network
        network.train(&train_inputs, &train_targets, 200);
        
        // Evaluate performance
        let train_acc = network.evaluate(&train_inputs, &train_targets, 0.5);
        let test_acc = network.evaluate(&test_inputs, &test_targets, 0.5);
        
        // Network should learn something (> 50% random chance)
        assert!(train_acc > 0.5, "Training accuracy {} too low", train_acc);
        assert!(test_acc > 0.4, "Test accuracy {} too low", test_acc);
        
        // Test prediction on a clearly fraudulent sample
        let high_fraud_features = Matrix::from_vec(15, 1, vec![
            0.1, 0.9, 0.9,  // Temporal: suspicious
            0.9, 0.9, 0.9, 0.9,  // Identity: many red flags
            0.9, 0.2, 0.9, 0.9,  // Employment: issues
            0.9, 0.9,  // Geographic: mismatches
            0.9, 0.9,  // Behavioral: suspicious
        ]);
        
        let fraud_prediction = network.predict(&high_fraud_features);
        let fraud_score = fraud_prediction.get(0, 0);
        
        // Should produce some output in valid range
        assert!(fraud_score >= 0.0 && fraud_score <= 1.0, 
            "Fraud score {} out of range", fraud_score);
    }

    #[test]
    fn test_fraud_feature_distinctiveness() {
        use crate::data::FraudDataGenerator;
        
        let mut fraud_gen = FraudDataGenerator::new(999);
        
        // Generate samples with different fraud ratios
        let (all_legit_inputs, all_legit_targets) = fraud_gen.generate_fraud_data(50, 0.0);
        let (all_fraud_inputs, all_fraud_targets) = fraud_gen.generate_fraud_data(50, 1.0);
        
        // Verify correct labels
        for target in &all_legit_targets {
            assert_eq!(target.get(0, 0), 0.0);
        }
        
        for target in &all_fraud_targets {
            assert_eq!(target.get(0, 0), 1.0);
        }
        
        // Calculate mean feature values for each class
        let mut legit_means = vec![0.0; 15];
        let mut fraud_means = vec![0.0; 15];
        
        for input in &all_legit_inputs {
            for i in 0..15 {
                legit_means[i] += input.get(i, 0);
            }
        }
        
        for input in &all_fraud_inputs {
            for i in 0..15 {
                fraud_means[i] += input.get(i, 0);
            }
        }
        
        for i in 0..15 {
            legit_means[i] /= all_legit_inputs.len() as f64;
            fraud_means[i] /= all_fraud_inputs.len() as f64;
        }
        
        // Key fraud indicators should have higher means in fraud samples
        // Feature 1: Claim frequency
        // Feature 4: SSN reuse
        // Feature 6: Address changes
        assert!(fraud_means[1] > legit_means[1], 
            "Fraud claim frequency should be higher");
        assert!(fraud_means[4] > legit_means[4], 
            "Fraud SSN reuse should be higher");
    }

    #[test]
    fn test_cnn_architecture() {
        // Create a simple MNIST CNN
        let mut network = MnistCNN::new(0.01, 42);
        
        // Create a test input (28x28 = 784 pixels)
        let input_data: Vec<f64> = (0..784).map(|i| (i as f64) / 784.0).collect();
        let input = Matrix::from_vec(784, 1, input_data);
        
        // Forward pass
        let output = network.predict(&input);
        
        // Check output shape
        assert_eq!(output.rows, 10);
        assert_eq!(output.cols, 1);
        
        // Check that output is a valid probability distribution (sums to ~1.0)
        let sum: f64 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Softmax output should sum to 1.0, got {}", sum);
        
        // Check that all outputs are between 0 and 1
        for i in 0..10 {
            let val = output.get(i, 0);
            assert!(val >= 0.0 && val <= 1.0, "Output {} should be in [0, 1], got {}", i, val);
        }
    }

    #[test]
    fn test_cnn_training_step() {
        // Create a simple MNIST CNN
        let mut network = MnistCNN::new(0.01, 42);
        
        // Create a test input
        let input_data: Vec<f64> = (0..784).map(|i| (i as f64) / 784.0).collect();
        let input = Matrix::from_vec(784, 1, input_data);
        
        // Create a target (one-hot encoding for digit 5)
        let mut target_data = vec![0.0; 10];
        target_data[5] = 1.0;
        let target = Matrix::from_vec(10, 1, target_data);
        
        // Training step
        let loss = network.train_step(&input, &target);
        
        // Check that loss is a reasonable positive number
        assert!(loss > 0.0, "Loss should be positive");
        assert!(loss < 10.0, "Loss should be reasonable");
    }

    #[test]
    fn test_cnn_batch_norm_modes() {
        // Create a CNN
        let mut network = MnistCNN::new(0.01, 42);
        
        // Create a test input
        let input_data: Vec<f64> = (0..784).map(|i| (i as f64) / 784.0).collect();
        let input = Matrix::from_vec(784, 1, input_data);
        
        // Test in training mode
        network.set_training(true);
        let output1 = network.predict(&input);
        
        // Test in inference mode
        network.set_training(false);
        let output2 = network.predict(&input);
        
        // Both outputs should be valid (10 classes, sum to ~1.0)
        assert_eq!(output1.rows, 10);
        assert_eq!(output2.rows, 10);
        
        let sum1: f64 = output1.data.iter().sum();
        let sum2: f64 = output2.data.iter().sum();
        
        assert!((sum1 - 1.0).abs() < 0.001);
        assert!((sum2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cnn_json_export() {
        // Create a CNN
        let network = MnistCNN::new(0.01, 42);
        
        // Export to JSON string
        let json = network.to_json_str().expect("Failed to export CNN to JSON");
        
        // Parse to verify structure
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse JSON");
        
        // Verify key fields exist
        assert_eq!(parsed["conv_in_channels"], 1);
        assert_eq!(parsed["conv_out_channels"], 16);
        assert_eq!(parsed["conv_kernel_size"], 5);
        assert_eq!(parsed["bn_num_features"], 16);
        assert_eq!(parsed["pool_kernel_size"], 2);
        assert_eq!(parsed["fc_weights"]["rows"], 10);
        assert_eq!(parsed["fc_weights"]["cols"], 3136);
    }
}
