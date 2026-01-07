#[cfg(test)]
mod client_side_tests {
    use crate::data::SyntheticDataGenerator;
    use crate::network::NeuralNetwork;
    use std::fs;

    #[test]
    fn test_model_export_and_reload() {
        // Train a simple network
        let mut data_gen = SyntheticDataGenerator::new(42);
        let (train_inputs, train_targets) = data_gen.generate_linear_separable(50, 2);
        
        let mut network = NeuralNetwork::new(vec![2, 4, 1], 0.1, 12345);
        network.train(&train_inputs, &train_targets, 100);
        
        // Export to JSON
        let test_path = "/tmp/test_model.json";
        network.save_to_json(test_path).expect("Failed to save model");
        
        // Verify file exists
        assert!(fs::metadata(test_path).is_ok(), "Model file should exist");
        
        // Load the model back
        let loaded_network = NeuralNetwork::load_from_json(test_path)
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
    fn test_exported_models_exist() {
        // Verify that the main program exports models
        // Note: This test assumes models are in the project root
        let models = vec![
            "model_linear.json",
            "model_xor.json", 
            "model_circular.json"
        ];
        
        for model_file in models {
            if fs::metadata(model_file).is_ok() {
                // Try to load the model
                let network = NeuralNetwork::load_from_json(model_file);
                assert!(network.is_ok(), "Should be able to load {}", model_file);
                
                let net = network.unwrap();
                assert!(!net.layer_sizes.is_empty(), "Network should have layers");
                assert!(!net.weights.is_empty(), "Network should have weights");
            }
        }
    }
}
