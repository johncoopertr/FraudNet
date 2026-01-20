#!/usr/bin/env python3
"""
Test the MNIST model with synthetic digit images.
This demonstrates that the model works correctly without requiring a webcam.
"""

import numpy as np
import json
import sys
from pathlib import Path

# Load the sklearn digits dataset for testing
try:
    from sklearn.datasets import load_digits
    import pandas as pd
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "pandas"])
    from sklearn.datasets import load_digits
    import pandas as pd

def load_mnist_model(json_path):
    """Load model from JSON."""
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    return model_data

def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def predict(model, input_pixels):
    """Run forward pass through the neural network."""
    # Start with input
    activation = np.array(input_pixels).reshape(-1, 1)
    
    weights = model['weights']
    biases = model['biases']
    
    # Forward pass through each layer
    for i in range(len(weights)):
        # Get weight and bias matrices
        W = np.array(weights[i]['data']).reshape(weights[i]['rows'], weights[i]['cols'])
        b = np.array(biases[i]['data']).reshape(-1, 1)
        
        # Linear transformation
        z = np.dot(W, activation) + b
        
        # Apply activation function
        if i < len(weights) - 1:
            # Hidden layers use ReLU
            activation = relu(z)
        else:
            # Output layer uses Sigmoid
            activation = sigmoid(z)
    
    return activation.flatten()

def test_mnist_model():
    """Test the MNIST model with real digit images."""
    print("=" * 70)
    print("MNIST Model Testing")
    print("=" * 70)
    
    # Load the trained model
    model_path = Path(__file__).parent.parent / 'model_mnist.json'
    print(f"\nLoading model from {model_path}...")
    model = load_mnist_model(model_path)
    
    print(f"  Architecture: {' → '.join(map(str, model['layer_sizes']))}")
    print(f"  Learning Rate: {model['learning_rate']}")
    print(f"  Layers: {len(model['weights'])}")
    
    # Load test digits
    print("\nLoading test digits from sklearn...")
    digits = load_digits()
    
    # Test on first 100 digits
    n_test = 100
    # Maximum pixel value in sklearn's digits dataset (8-bit grayscale, max value is 16)
    SKLEARN_DIGITS_MAX_VALUE = 16.0
    print(f"\nTesting on {n_test} digit images...\n")
    
    correct = 0
    predictions_by_digit = {i: {'correct': 0, 'total': 0} for i in range(10)}
    
    for i in range(n_test):
        # Get the 8x8 image and resize to 28x28
        img_8x8 = digits.images[i]
        true_label = digits.target[i]
        
        # Simple upscaling to 28x28
        img_28x28 = np.zeros((28, 28))
        upscaled = np.repeat(np.repeat(img_8x8, 2, axis=0), 2, axis=1)
        
        # Place in center
        start = (28 - 16) // 2
        img_28x28[start:start+16, start:start+16] = upscaled
        
        # Normalize to [0, 1]
        pixels = (img_28x28 / SKLEARN_DIGITS_MAX_VALUE).flatten()
        
        # Run prediction
        output = predict(model, pixels)
        predicted_label = np.argmax(output)
        confidence = output[predicted_label] * 100
        
        # Track results
        predictions_by_digit[true_label]['total'] += 1
        if predicted_label == true_label:
            correct += 1
            predictions_by_digit[true_label]['correct'] += 1
            status = "✓"
        else:
            status = "✗"
        
        if i < 10:  # Show first 10 predictions
            print(f"  {status} Sample {i+1}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.1f}%")
    
    accuracy = (correct / n_test) * 100
    
    print(f"\n{'-' * 70}")
    print(f"Overall Accuracy: {correct}/{n_test} = {accuracy:.2f}%")
    print(f"{'-' * 70}")
    
    print("\nPer-Digit Accuracy:")
    for digit in range(10):
        stats = predictions_by_digit[digit]
        if stats['total'] > 0:
            digit_acc = (stats['correct'] / stats['total']) * 100
            print(f"  Digit {digit}: {stats['correct']}/{stats['total']} = {digit_acc:.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ Model testing complete!")
    print("=" * 70)
    
    return accuracy >= 85  # Pass if accuracy is at least 85%

if __name__ == '__main__':
    success = test_mnist_model()
    sys.exit(0 if success else 1)
