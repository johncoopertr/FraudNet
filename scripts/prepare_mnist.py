#!/usr/bin/env python3
"""
Download and prepare the real MNIST dataset in parquet format.

This script downloads the official MNIST dataset and converts it to
parquet format for use with the Rust neural network trainer.

If network access is restricted, it will augment existing data to create
a larger training set.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def download_mnist():
    """Download the real MNIST dataset using keras/tensorflow."""
    try:
        from tensorflow import keras
        print("Downloading MNIST dataset from Keras...")
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        
    except (ImportError, Exception) as e:
        print(f"TensorFlow download failed: {e}")
        print("Trying torchvision...")
        try:
            import torchvision
            from torchvision import datasets, transforms
            
            # Download to temporary directory
            mnist_train = datasets.MNIST(root='./tmp', train=True, download=True)
            mnist_test = datasets.MNIST(root='./tmp', train=False, download=True)
            
            train_images = mnist_train.data.numpy()
            train_labels = mnist_train.targets.numpy()
            test_images = mnist_test.data.numpy()
            test_labels = mnist_test.targets.numpy()
            
        except (ImportError, Exception) as e2:
            print(f"PyTorch download failed: {e2}")
            print("Network access is restricted. Using data augmentation to expand existing dataset...")
            return augment_existing_data()
    
    print(f"  Training images: {train_images.shape}")
    print(f"  Training labels: {train_labels.shape}")
    print(f"  Test images: {test_images.shape}")
    print(f"  Test labels: {test_labels.shape}")
    
    # Normalize to [0, 1] range
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # Flatten images to 784 features
    train_images = train_images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)
    
    return train_images, train_labels, test_images, test_labels


def augment_existing_data():
    """Augment existing parquet data to create a larger dataset."""
    project_dir = Path(__file__).parent.parent
    train_path = project_dir / 'mnist-train.parquet'
    test_path = project_dir / 'mnist-test.parquet'
    
    if not train_path.exists() or not test_path.exists():
        print("No existing data found. Generating synthetic data...")
        return generate_mnist_like_data()
    
    print("Loading existing data for augmentation...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Extract images and labels
    train_images_2d = train_df.iloc[:, :-1].values
    train_labels = train_df.iloc[:, -1].values
    test_images_2d = test_df.iloc[:, :-1].values
    test_labels = test_df.iloc[:, -1].values
    
    # Reshape to 28x28 for augmentation
    train_images_28x28 = train_images_2d.reshape(-1, 28, 28)
    test_images_28x28 = test_images_2d.reshape(-1, 28, 28)
    
    # Target sizes
    target_train_size = 60000
    target_test_size = 10000
    
    # Augment training data
    print(f"Augmenting training data from {len(train_images_28x28)} to {target_train_size}...")
    augmented_train_images = []
    augmented_train_labels = []
    
    augmentation_factor = target_train_size // len(train_images_28x28)
    np.random.seed(42)
    
    for i in range(len(train_images_28x28)):
        for aug_idx in range(augmentation_factor):
            img = train_images_28x28[i].copy()
            
            if aug_idx == 0:
                # Original image
                augmented_train_images.append(img.flatten())
            else:
                # Apply augmentations
                # 1. Small rotation (±5 degrees)
                angle = np.random.uniform(-5, 5)
                img = rotate_image(img, angle)
                
                # 2. Small translation (±2 pixels)
                dx = np.random.randint(-2, 3)
                dy = np.random.randint(-2, 3)
                img = translate_image(img, dx, dy)
                
                # 3. Small noise
                noise = np.random.normal(0, 0.02, img.shape)
                img = np.clip(img + noise, 0, 1)
                
                augmented_train_images.append(img.flatten())
            
            augmented_train_labels.append(train_labels[i])
    
    # If we need more samples, repeat with more augmentation
    while len(augmented_train_images) < target_train_size:
        idx = np.random.randint(0, len(train_images_28x28))
        img = train_images_28x28[idx].copy()
        
        # Apply random augmentations
        angle = np.random.uniform(-10, 10)
        img = rotate_image(img, angle)
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        img = translate_image(img, dx, dy)
        noise = np.random.normal(0, 0.03, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        augmented_train_images.append(img.flatten())
        augmented_train_labels.append(train_labels[idx])
    
    # Augment test data similarly
    print(f"Augmenting test data from {len(test_images_28x28)} to {target_test_size}...")
    augmented_test_images = []
    augmented_test_labels = []
    
    test_augmentation_factor = target_test_size // len(test_images_28x28)
    
    for i in range(len(test_images_28x28)):
        for aug_idx in range(test_augmentation_factor):
            img = test_images_28x28[i].copy()
            
            if aug_idx == 0:
                augmented_test_images.append(img.flatten())
            else:
                angle = np.random.uniform(-3, 3)
                img = rotate_image(img, angle)
                dx = np.random.randint(-1, 2)
                dy = np.random.randint(-1, 2)
                img = translate_image(img, dx, dy)
                noise = np.random.normal(0, 0.01, img.shape)
                img = np.clip(img + noise, 0, 1)
                augmented_test_images.append(img.flatten())
            
            augmented_test_labels.append(test_labels[i])
    
    while len(augmented_test_images) < target_test_size:
        idx = np.random.randint(0, len(test_images_28x28))
        img = test_images_28x28[idx].copy()
        angle = np.random.uniform(-5, 5)
        img = rotate_image(img, angle)
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(-2, 3)
        img = translate_image(img, dx, dy)
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)
        augmented_test_images.append(img.flatten())
        augmented_test_labels.append(test_labels[idx])
    
    # Convert to numpy arrays
    train_images = np.array(augmented_train_images[:target_train_size], dtype=np.float32)
    train_labels = np.array(augmented_train_labels[:target_train_size])
    test_images = np.array(augmented_test_images[:target_test_size], dtype=np.float32)
    test_labels = np.array(augmented_test_labels[:target_test_size])
    
    print(f"  Augmented training images: {train_images.shape}")
    print(f"  Augmented training labels: {train_labels.shape}")
    print(f"  Augmented test images: {test_images.shape}")
    print(f"  Augmented test labels: {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels


def rotate_image(img, angle):
    """Rotate image by angle degrees."""
    from scipy import ndimage
    return ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0)


def translate_image(img, dx, dy):
    """Translate image by dx, dy pixels."""
    from scipy import ndimage
    return ndimage.shift(img, [dy, dx], mode='constant', cval=0)


def generate_mnist_like_data():
    """Generate MNIST-like dataset using sklearn's digits dataset (8x8)
    and synthetic data to create a full 28x28 dataset."""
    
    print("Generating MNIST-like dataset...")
    from sklearn.datasets import load_digits
    
    # Load sklearn's built-in digits dataset (8x8, 1797 samples)
    digits = load_digits()
    
    # We'll create synthetic data based on this
    # Create training set: 60000 samples
    # Create test set: 10000 samples
    
    n_train = 60000
    n_test = 10000
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
    
    # Download real MNIST data (or augment existing data if download fails)
    train_images, train_labels, test_images, test_labels = download_mnist()
    
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
    print("MNIST Dataset Summary")
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
