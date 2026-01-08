mod activation;
mod data;
mod matrix;
mod model_export;
mod network;
mod tests;
mod utils;

use data::SyntheticDataGenerator;
use network::NeuralNetwork;

fn main() {
    println!("FraudNet - Neural Network Classifier");
    println!("=====================================\n");

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
    println!("\nExporting trained models to JSON...");
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
    println!("\nConverting models to ONNX format...");
    println!("Run: python3 scripts/json_to_onnx.py");
    println!("This will create .onnx files for client-side inference with ONNX Runtime.");
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
