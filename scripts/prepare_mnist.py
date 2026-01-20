#!/usr/bin/env python3
"""
Generate synthetic MNIST-like dataset in parquet format.

This script generates synthetic handwritten digit data similar to MNIST
for use with the Rust neural network trainer.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_digits


def generate_mnist_like_data():
    """Generate MNIST-like dataset using sklearn's digits dataset (8x8)
    and synthetic data to create a full 28x28 dataset."""
    
    print("Generating MNIST-like dataset...")
    
    # Load sklearn's built-in digits dataset (8x8, 1797 samples)
    digits = load_digits()
    
    # We'll create synthetic data based on this
    # Create training set: 6000 samples
    # Create test set: 1000 samples
    
    n_train = 6000
    n_test = 1000
    n_features = 784  # 28x28 pixels
    
    np.random.seed(42)
    
    # Generate training data
    print("\nGenerating training data...")
    train_images = []
    train_labels = []
    
    for i in range(n_train):
        # Pick a random digit from sklearn dataset
        idx = np.random.randint(0, len(digits.data))
        label = digits.target[idx]
        
        # Create a 28x28 image with the 8x8 digit in the center
        # surrounded by noise
        image = np.zeros((28, 28), dtype=np.float32)
        
        # Add some background noise
        image += np.random.normal(0, 0.05, (28, 28)).astype(np.float32)
        
        # Resize the 8x8 digit to approximately 20x20 and place in center
        digit_img = digits.images[idx].astype(np.float32) / 16.0  # Normalize to [0, 1]
        
        # Simple upscaling: repeat pixels
        upscaled = np.repeat(np.repeat(digit_img, 2, axis=0), 2, axis=1)
        # Add some variation
        upscaled = upscaled + np.random.normal(0, 0.02, upscaled.shape).astype(np.float32)
        
        # Place in center of 28x28 image
        start_y = (28 - 16) // 2
        start_x = (28 - 16) // 2
        image[start_y:start_y+16, start_x:start_x+16] = upscaled
        
        # Clip to [0, 1] range
        image = np.clip(image, 0, 1)
        
        # Flatten to 784 features
        train_images.append(image.flatten())
        train_labels.append(label)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    print(f"  Training images: {train_images.shape}")
    print(f"  Training labels: {train_labels.shape}")
    
    # Generate test data
    print("\nGenerating test data...")
    test_images = []
    test_labels = []
    
    for i in range(n_test):
        idx = np.random.randint(0, len(digits.data))
        label = digits.target[idx]
        
        image = np.zeros((28, 28), dtype=np.float32)
        image += np.random.normal(0, 0.05, (28, 28)).astype(np.float32)
        
        digit_img = digits.images[idx].astype(np.float32) / 16.0
        upscaled = np.repeat(np.repeat(digit_img, 2, axis=0), 2, axis=1)
        upscaled = upscaled + np.random.normal(0, 0.02, upscaled.shape).astype(np.float32)
        
        start_y = (28 - 16) // 2
        start_x = (28 - 16) // 2
        image[start_y:start_y+16, start_x:start_x+16] = upscaled
        
        image = np.clip(image, 0, 1)
        
        test_images.append(image.flatten())
        test_labels.append(label)
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    print(f"  Test images: {test_images.shape}")
    print(f"  Test labels: {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels


def create_parquet_files():
    """Create parquet files for MNIST training and test sets."""
    
    # Generate data
    train_images, train_labels, test_images, test_labels = generate_mnist_like_data()
    
    # Create DataFrames
    # Each row has 784 pixel features + 1 label column
    train_df = pd.DataFrame(train_images)
    train_df['label'] = train_labels
    
    test_df = pd.DataFrame(test_images)
    test_df['label'] = test_labels
    
    # Save to parquet
    project_dir = Path(__file__).parent.parent
    train_path = project_dir / 'mnist-train.parquet'
    test_path = project_dir / 'mnist-test.parquet'
    
    print("\nSaving parquet files...")
    train_df.to_parquet(train_path, compression='snappy', index=False)
    print(f"  ✓ Saved training data to {train_path}")
    print(f"    Size: {train_path.stat().st_size / (1024*1024):.2f} MB")
    
    test_df.to_parquet(test_path, compression='snappy', index=False)
    print(f"  ✓ Saved test data to {test_path}")
    print(f"    Size: {test_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Print data summary
    print("\n" + "="*50)
    print("MNIST-like Dataset Summary")
    print("="*50)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Input features: 784 (28x28 pixels)")
    print(f"Output classes: 10 (digits 0-9)")
    print(f"\nLabel distribution (training):")
    print(train_df['label'].value_counts().sort_index())
    print("="*50)


if __name__ == '__main__':
    create_parquet_files()
