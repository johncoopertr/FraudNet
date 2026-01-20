use fraudnet::matrix::Matrix;
use fraudnet::cnn::MnistCNN;
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

fn argmax(values: &[f64]) -> usize {
    let mut max_idx = 0;
    let mut max_val = values[0];
    for i in 1..values.len() {
        if values[i] > max_val {
            max_val = values[i];
            max_idx = i;
        }
    }
    max_idx
}

fn evaluate_mnist(network: &mut MnistCNN, inputs: &[Matrix], targets: &[Matrix]) -> f64 {
    let mut correct = 0;
    
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let prediction = network.predict(input);
        
        // Find predicted class (max output)
        let mut pred_values = Vec::with_capacity(10);
        for i in 0..10 {
            pred_values.push(prediction.get(i, 0));
        }
        let pred_class = argmax(&pred_values);
        
        // Find true class
        let mut true_values = Vec::with_capacity(10);
        for i in 0..10 {
            true_values.push(target.get(i, 0));
        }
        let true_class = argmax(&true_values);
        
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
    
    // Create MNIST CNN architecture
    let learning_rate = 0.01;
    let mut network = MnistCNN::new(learning_rate, 42);
    
    println!("CNN Architecture:");
    println!("  Layer 1: ZeroPad2d       - 28x28 -> 32x32");
    println!("  Layer 2: Conv2d          - 16 filters, 5x5 kernel, stride 1");
    println!("  Layer 3: BatchNorm2d     - 16 features");
    println!("  Layer 4: ReLU            - activation");
    println!("  Layer 5: MaxPool2d       - 2x2 kernel, stride 2");
    println!("  Layer 6: Flatten         - 3136 features (16*14*14)");
    println!("  Layer 7: Linear          - 3136 -> 10");
    println!("  Layer 8: Softmax         - 10 classes");
    println!("  Learning Rate:  {}", learning_rate);
    println!();
    
    // Train the network
    let epochs = 10;
    println!("Training for {} epochs...", epochs);
    println!("(This may take a few minutes)\n");
    
    for epoch in 0..epochs {
        network.set_training(true);
        let mut total_loss = 0.0;
        
        for (input, target) in train_inputs.iter().zip(train_targets.iter()) {
            let loss = network.train_step(input, target);
            total_loss += loss;
        }
        
        let avg_loss = total_loss / train_inputs.len() as f64;
        
        // Evaluate accuracy every epoch
        network.set_training(false);
        let train_acc = evaluate_mnist(&mut network, &train_inputs, &train_targets);
        let test_acc = evaluate_mnist(&mut network, &test_inputs, &test_targets);
        
        println!(
            "Epoch {}: Loss = {:.6}, Train Acc = {:.2}%, Test Acc = {:.2}%",
            epoch, avg_loss, train_acc * 100.0, test_acc * 100.0
        );
    }
    
    // Final evaluation
    network.set_training(false);
    println!("\n{}", "=".repeat(50));
    println!("Final Performance:");
    let train_acc = evaluate_mnist(&mut network, &train_inputs, &train_targets);
    let test_acc = evaluate_mnist(&mut network, &test_inputs, &test_targets);
    println!("  Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("  Testing Accuracy:  {:.2}%", test_acc * 100.0);
    println!("{}", "=".repeat(50));
    
    // Test on a few examples
    println!("\nExample Predictions:");
    for i in 0..5 {
        let prediction = network.predict(&test_inputs[i]);
        
        // Find predicted class using argmax helper
        let mut pred_values = Vec::with_capacity(10);
        for j in 0..10 {
            pred_values.push(prediction.get(j, 0));
        }
        let pred_class = argmax(&pred_values);
        let max_val = pred_values[pred_class];
        
        // Find true class using argmax helper
        let mut true_values = Vec::with_capacity(10);
        for j in 0..10 {
            true_values.push(test_targets[i].get(j, 0));
        }
        let true_class = argmax(&true_values);
        
        println!(
            "  Sample {}: Predicted = {}, True = {}, Confidence = {:.2}%",
            i + 1, pred_class, true_class, max_val * 100.0
        );
    }
    
    println!("\nCNN Training Complete!");
    println!("Note: Full model export to ONNX not yet implemented for CNN architecture.");
    
    Ok(())
}
