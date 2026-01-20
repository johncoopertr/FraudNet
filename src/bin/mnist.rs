use fraudnet::matrix::Matrix;
use fraudnet::network::NeuralNetwork;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;

fn load_mnist_data(path: &str) -> Result<(Vec<Matrix>, Vec<Matrix>), Box<dyn std::error::Error>> {
    println!("Loading MNIST data from {}...", path);
    
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    let row_iter = reader.get_row_iter(None)?;
    
    for record in row_iter {
        let row = record?;
        
        // Extract 784 pixel features (0-783)
        let mut pixels = Vec::with_capacity(784);
        for i in 0..784 {
            let pixel_value = row.get_float(i)? as f64;
            pixels.push(pixel_value);
        }
        
        // Extract label (column 784)
        let label = row.get_long(784)? as usize;
        
        // Create one-hot encoded target vector for 10 classes
        let mut target_vec = vec![0.0; 10];
        target_vec[label] = 1.0;
        
        // Create matrices
        let input = Matrix::from_vec(784, 1, pixels);
        let target = Matrix::from_vec(10, 1, target_vec);
        
        inputs.push(input);
        targets.push(target);
    }
    
    println!("  Loaded {} samples", inputs.len());
    Ok((inputs, targets))
}

fn evaluate_mnist(network: &NeuralNetwork, inputs: &[Matrix], targets: &[Matrix]) -> f64 {
    let mut correct = 0;
    
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let prediction = network.predict(input);
        
        // Find predicted class (max output)
        let mut pred_class = 0;
        let mut max_val = prediction.get(0, 0);
        for i in 1..10 {
            let val = prediction.get(i, 0);
            if val > max_val {
                max_val = val;
                pred_class = i;
            }
        }
        
        // Find true class
        let mut true_class = 0;
        for i in 0..10 {
            if target.get(i, 0) > 0.5 {
                true_class = i;
                break;
            }
        }
        
        if pred_class == true_class {
            correct += 1;
        }
    }
    
    correct as f64 / inputs.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MNIST Digit Recognition Neural Network");
    println!("======================================\n");
    
    // Load MNIST training data from parquet
    let (train_inputs, train_targets) = load_mnist_data("mnist-train.parquet")?;
    
    // Load MNIST test data from parquet
    let (test_inputs, test_targets) = load_mnist_data("mnist-test.parquet")?;
    
    println!("\nDataset Information:");
    println!("  Training samples: {}", train_inputs.len());
    println!("  Testing samples:  {}", test_inputs.len());
    println!("  Input features:   784 (28x28 pixels)");
    println!("  Output classes:   10 (digits 0-9)");
    println!();
    
    // Create MNIST network architecture
    // 784 inputs -> 128 hidden -> 64 hidden -> 10 outputs
    let architecture = vec![784, 128, 64, 10];
    let learning_rate = 0.01;
    let mut network = NeuralNetwork::new(architecture.clone(), learning_rate, 42);
    
    println!("Network Architecture:");
    println!("  Input Layer:    784 neurons (28x28 pixels)");
    println!("  Hidden Layer 1: 128 neurons (ReLU)");
    println!("  Hidden Layer 2: 64 neurons (ReLU)");
    println!("  Output Layer:   10 neurons (Sigmoid) - digit classes 0-9");
    println!("  Learning Rate:  {}", learning_rate);
    println!();
    
    // Train the network
    let epochs = 10;
    println!("Training for {} epochs...", epochs);
    println!("(This may take a few minutes)\n");
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        
        for (input, target) in train_inputs.iter().zip(train_targets.iter()) {
            let loss = network.train_step(input, target);
            total_loss += loss;
        }
        
        let avg_loss = total_loss / train_inputs.len() as f64;
        
        // Evaluate accuracy every epoch
        let train_acc = evaluate_mnist(&network, &train_inputs, &train_targets);
        let test_acc = evaluate_mnist(&network, &test_inputs, &test_targets);
        
        println!(
            "Epoch {}: Loss = {:.6}, Train Acc = {:.2}%, Test Acc = {:.2}%",
            epoch, avg_loss, train_acc * 100.0, test_acc * 100.0
        );
    }
    
    // Final evaluation
    println!("\n{}", "=".repeat(50));
    println!("Final Performance:");
    let train_acc = evaluate_mnist(&network, &train_inputs, &train_targets);
    let test_acc = evaluate_mnist(&network, &test_inputs, &test_targets);
    println!("  Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("  Testing Accuracy:  {:.2}%", test_acc * 100.0);
    println!("{}", "=".repeat(50));
    
    // Test on a few examples
    println!("\nExample Predictions:");
    for i in 0..5 {
        let prediction = network.predict(&test_inputs[i]);
        
        // Find predicted class
        let mut pred_class = 0;
        let mut max_val = prediction.get(0, 0);
        for j in 1..10 {
            let val = prediction.get(j, 0);
            if val > max_val {
                max_val = val;
                pred_class = j;
            }
        }
        
        // Find true class
        let mut true_class = 0;
        for j in 0..10 {
            if test_targets[i].get(j, 0) > 0.5 {
                true_class = j;
                break;
            }
        }
        
        println!(
            "  Sample {}: Predicted = {}, True = {}, Confidence = {:.2}%",
            i + 1, pred_class, true_class, max_val * 100.0
        );
    }
    
    // Export trained model to JSON
    println!("\nExporting trained model...");
    network.save_to_json("model_mnist.json")?;
    println!("  ✓ Saved model_mnist.json");
    
    println!("\nNext Steps:");
    println!("  1. Run: python3 scripts/json_to_onnx.py");
    println!("     This will convert model_mnist.json to model_mnist.onnx");
    println!("  2. Open web/mnist.html in a browser to test webcam digit recognition");
    
    Ok(())
}
