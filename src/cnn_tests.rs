use fraudnet::matrix::Matrix;
use fraudnet::cnn::MnistCNN;

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
