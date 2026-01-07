mod activation;
mod data;
mod matrix;
mod network;
mod tests;

use data::SyntheticDataGenerator;
use network::NeuralNetwork;

fn main() {
    println!("FraudNet - Neural Network Classifier");
    println!("=====================================\n");

    // Test 1: Simple linearly separable data
    println!("Test 1: Linearly Separable Data");
    println!("--------------------------------");
    test_linear_separable();

    println!("\n");

    // Test 2: XOR problem (non-linearly separable)
    println!("Test 2: XOR Problem (Non-linear)");
    println!("---------------------------------");
    test_xor();

    println!("\n");

    // Test 3: Circular boundary (non-linearly separable)
    println!("Test 3: Circular Boundary (Non-linear)");
    println!("---------------------------------------");
    test_circular();
}

fn test_linear_separable() {
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
}

fn test_xor() {
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
}

fn test_circular() {
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
}
