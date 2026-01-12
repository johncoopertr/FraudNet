mod activation;
mod data;
mod matrix;
mod model_export;
mod network;
mod tests;
mod utils;
mod db_schema;
mod db_loader;

use data::{SyntheticDataGenerator, FraudDataGenerator};
use matrix::Matrix;
use network::NeuralNetwork;
use db_loader::{DatabaseConfig, load_from_database, split_data, records_to_training_data, export_demo_data};

fn main() {
    // Load environment variables from .env file if it exists
    // Silently ignoring errors allows the program to work without a .env file
    // and fall back to synthetic data generation
    dotenv::dotenv().ok();
    
    println!("FraudNet - Unemployment Insurance Fraud Detection");
    println!("=================================================\n");

    // Primary use case: Unemployment Insurance Fraud Detection
    println!("Primary Model: Unemployment Insurance Fraud Detection");
    println!("-----------------------------------------------------");
    let fraud_network = test_unemployment_fraud_detection();

    println!("\n\n");
    println!("Additional Test Models (for validation)");
    println!("========================================\n");

    // Test 1: Simple linearly separable data
    println!("Test 1: Linearly Separable Data");
    println!("--------------------------------");
    let network1 = test_linear_separable();

    println!("\n");

    // Test 2: XOR problem (non-linearly separable)
    println!("Test 2: XOR Problem (Non-linear)");
    println!("---------------------------------");
    let network2 = test_xor();

    println!("\n");

    // Test 3: Circular boundary (non-linearly separable)
    println!("Test 3: Circular Boundary (Non-linear)");
    println!("---------------------------------------");
    let network3 = test_circular();
    
    // Export trained models to JSON
    println!("\n\nExporting trained models to JSON...");
    println!("===================================");
    
    if let Err(e) = fraud_network.save_to_json("model_fraud_detection.json") {
        eprintln!("Failed to export fraud detection model: {}", e);
    } else {
        println!("✓ Exported model_fraud_detection.json (PRIMARY MODEL)");
    }
    
    if let Err(e) = network1.save_to_json("model_linear.json") {
        eprintln!("Failed to export linear model: {}", e);
    } else {
        println!("✓ Exported model_linear.json");
    }
    
    if let Err(e) = network2.save_to_json("model_xor.json") {
        eprintln!("Failed to export XOR model: {}", e);
    } else {
        println!("✓ Exported model_xor.json");
    }
    
    if let Err(e) = network3.save_to_json("model_circular.json") {
        eprintln!("Failed to export circular model: {}", e);
    } else {
        println!("✓ Exported model_circular.json");
    }
    
    // Convert JSON models to ONNX format
    println!("\n\nNext Steps");
    println!("==========");
    println!("Run: python3 scripts/json_to_onnx.py");
    println!("This will create .onnx files for client-side inference with ONNX Runtime.");
    println!("\nThe fraud detection model can be deployed to detect unemployment insurance fraud in real-time.");
    println!("See FRAUD_DETECTION.md for detailed documentation on features and usage.");
}

fn test_linear_separable() -> NeuralNetwork {
    let mut data_gen = SyntheticDataGenerator::new(42);

    // Generate training and testing data
    let (train_inputs, train_targets) = data_gen.generate_linear_separable(100, 3);
    let (test_inputs, test_targets) = data_gen.generate_linear_separable(50, 3);

    // Create a simple network: 3 inputs -> 8 hidden -> 1 output
    let mut network = NeuralNetwork::new(vec![3, 8, 1], 0.1, 12345);

    println!("Training network...");
    network.train(&train_inputs, &train_targets, 500);

    let train_acc = network.evaluate(&train_inputs, &train_targets, 0.5);
    let test_acc = network.evaluate(&test_inputs, &test_targets, 0.5);

    println!("\nResults:");
    println!("  Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("  Testing Accuracy:  {:.2}%", test_acc * 100.0);
    
    network
}

fn test_xor() -> NeuralNetwork {
    let mut data_gen = SyntheticDataGenerator::new(123);

    // Generate training and testing data
    let (train_inputs, train_targets) = data_gen.generate_xor(200);
    let (test_inputs, test_targets) = data_gen.generate_xor(100);

    // XOR requires hidden layer: 2 inputs -> 8 hidden -> 1 output
    let mut network = NeuralNetwork::new(vec![2, 8, 1], 0.1, 54321);

    println!("Training network...");
    network.train(&train_inputs, &train_targets, 1000);

    let train_acc = network.evaluate(&train_inputs, &train_targets, 0.5);
    let test_acc = network.evaluate(&test_inputs, &test_targets, 0.5);

    println!("\nResults:");
    println!("  Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("  Testing Accuracy:  {:.2}%", test_acc * 100.0);
    
    network
}

fn test_circular() -> NeuralNetwork {
    let mut data_gen = SyntheticDataGenerator::new(456);

    // Generate training and testing data
    let (train_inputs, train_targets) = data_gen.generate_circular(200, 0.5);
    let (test_inputs, test_targets) = data_gen.generate_circular(100, 0.5);

    // Circular boundary: 2 inputs -> 16 hidden -> 8 hidden -> 1 output
    let mut network = NeuralNetwork::new(vec![2, 16, 8, 1], 0.1, 98765);

    println!("Training network...");
    network.train(&train_inputs, &train_targets, 1000);

    let train_acc = network.evaluate(&train_inputs, &train_targets, 0.5);
    let test_acc = network.evaluate(&test_inputs, &test_targets, 0.5);

    println!("\nResults:");
    println!("  Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("  Testing Accuracy:  {:.2}%", test_acc * 100.0);
    
    network
}

fn test_unemployment_fraud_detection() -> NeuralNetwork {
    // Try to load real data from database if configured
    let use_real_data = match DatabaseConfig::from_env() {
        Ok(config) => {
            println!("Database configuration found. Attempting to load real data...");
            let demo_file = config.demo_output_file.clone();
            match load_from_database(&config) {
                Ok(records) => {
                    println!("✓ Successfully loaded {} records from database", records.len());
                    
                    // Split data into training, testing, and demo sets
                    let (train_records, test_records, demo_records) = split_data(records, &config);
                    
                    println!("\nData Split:");
                    println!("  Training records: {}", train_records.len());
                    println!("  Testing records:  {}", test_records.len());
                    println!("  Demo records:     {}", demo_records.len());
                    
                    // Export demo data for web demonstration
                    if !demo_records.is_empty() {
                        if let Err(e) = export_demo_data(&demo_records, &demo_file) {
                            eprintln!("Warning: Failed to export demo data: {}", e);
                        }
                    }
                    
                    // Convert to training format
                    let (train_inputs, train_targets) = records_to_training_data(&train_records);
                    let (test_inputs, test_targets) = records_to_training_data(&test_records);
                    
                    println!("\nDataset Information:");
                    println!("  Data source:      Real database records");
                    println!("  Training samples: {}", train_inputs.len());
                    println!("  Testing samples:  {}", test_inputs.len());
                    println!("  Input features:   15 (temporal, identity, employment, geographic, behavioral)");
                    println!();
                    
                    Some((train_inputs, train_targets, test_inputs, test_targets))
                }
                Err(e) => {
                    eprintln!("Failed to load from database: {}", e);
                    eprintln!("Falling back to synthetic data generation...\n");
                    None
                }
            }
        }
        Err(_) => {
            println!("No database configuration found (.env file not present)");
            println!("Using synthetic data for training and testing...\n");
            None
        }
    };
    
    // Use real data if available, otherwise generate synthetic data
    let (train_inputs, train_targets, test_inputs, test_targets) = match use_real_data {
        Some(data) => data,
        None => {
            let mut data_gen = FraudDataGenerator::new(42);
            
            // Generate training data: 1000 samples with 30% fraud rate
            let (train_inputs, train_targets) = data_gen.generate_fraud_data(1000, 0.30);
            
            // Generate testing data: 500 samples with 30% fraud rate
            let (test_inputs, test_targets) = data_gen.generate_fraud_data(500, 0.30);
            
            println!("Dataset Information:");
            println!("  Data source:      Synthetic data generation");
            println!("  Training samples: {} (70% legitimate, 30% fraudulent)", train_inputs.len());
            println!("  Testing samples:  {} (70% legitimate, 30% fraudulent)", test_inputs.len());
            println!("  Input features:   15 (temporal, identity, employment, geographic, behavioral)");
            println!();
            
            (train_inputs, train_targets, test_inputs, test_targets)
        }
    };

    // Deep network architecture for fraud detection
    // 15 inputs -> 9 hidden layers -> 1 output
    // This matches the architecture described in the README
    let architecture = vec![15, 64, 52, 42, 32, 26, 22, 20, 16, 8, 1];
    let mut network = NeuralNetwork::new(architecture, 0.08, 12345);

    println!("Network Architecture:");
    println!("  Input Layer:    15 features");
    println!("  Hidden Layer 1: 64 neurons (ReLU)");
    println!("  Hidden Layer 2: 52 neurons (ReLU)");
    println!("  Hidden Layer 3: 42 neurons (ReLU)");
    println!("  Hidden Layer 4: 32 neurons (ReLU)");
    println!("  Hidden Layer 5: 26 neurons (ReLU)");
    println!("  Hidden Layer 6: 22 neurons (ReLU)");
    println!("  Hidden Layer 7: 20 neurons (ReLU)");
    println!("  Hidden Layer 8: 16 neurons (ReLU)");
    println!("  Hidden Layer 9: 8 neurons (ReLU)");
    println!("  Output Layer:   1 neuron (Sigmoid) - Fraud probability");
    println!();

    println!("Training deep neural network for fraud detection...");
    println!("(This may take a few moments due to the deep architecture)");
    network.train(&train_inputs, &train_targets, 1500);

    let train_acc = network.evaluate(&train_inputs, &train_targets, 0.5);
    let test_acc = network.evaluate(&test_inputs, &test_targets, 0.5);

    println!("\nFraud Detection Performance:");
    println!("  Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("  Testing Accuracy:  {:.2}%", test_acc * 100.0);

    // Test on some example scenarios
    println!("\nExample Fraud Predictions:");
    println!("  (Threshold: 0.5 - scores >= 0.5 indicate likely fraud)\n");

    // Create a test legitimate claim
    let legitimate_sample = Matrix::from_vec(15, 1, vec![
        0.4, 0.1, 0.2,  // Temporal: normal timing
        0.05, 0.1, 0.1, 0.05,  // Identity: verified
        0.1, 0.6, 0.2, 0.2,  // Employment: good history
        0.1, 0.05,  // Geographic: consistent
        0.2, 0.1,  // Behavioral: normal
    ]);
    let legit_score = network.predict(&legitimate_sample).get(0, 0);
    println!("  Legitimate claim example:  {:.4} ({})", 
        legit_score, 
        if legit_score < 0.5 { "PASS" } else { "FLAG" }
    );

    // Create a test identity fraud claim
    let identity_fraud_sample = Matrix::from_vec(15, 1, vec![
        0.2, 0.8, 0.9,  // Temporal: suspicious
        0.9, 0.8, 0.7, 0.9,  // Identity: red flags
        0.4, 0.3, 0.5, 0.5,  // Employment: some issues
        0.6, 0.6,  // Geographic: mismatches
        0.7, 0.8,  // Behavioral: suspicious
    ]);
    let fraud_score = network.predict(&identity_fraud_sample).get(0, 0);
    println!("  Identity fraud example:    {:.4} ({})", 
        fraud_score,
        if fraud_score >= 0.5 { "FLAG" } else { "PASS" }
    );

    // Create a test concurrent employment fraud
    let concurrent_fraud_sample = Matrix::from_vec(15, 1, vec![
        0.3, 0.4, 0.2,  // Temporal: regular
        0.1, 0.1, 0.1, 0.1,  // Identity: legitimate person
        0.9, 0.1, 0.9, 0.8,  // Employment: hiding work
        0.2, 0.1,  // Geographic: consistent
        0.3, 0.7,  // Behavioral: evasive patterns
    ]);
    let concurrent_score = network.predict(&concurrent_fraud_sample).get(0, 0);
    println!("  Employment fraud example:  {:.4} ({})", 
        concurrent_score,
        if concurrent_score >= 0.5 { "FLAG" } else { "PASS" }
    );

    network
}
