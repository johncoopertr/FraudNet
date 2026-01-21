# MNIST Digit Recognition System

This document describes the MNIST digit recognition system implemented in FraudNet, which provides real-time optical character recognition (OCR) for handwritten digits using a webcam.

## Overview

The MNIST system is a complete end-to-end implementation that:
1. Trains a neural network on MNIST-like digit data
2. Exports the model to ONNX format for browser deployment
3. Provides a web interface for real-time digit recognition via webcam

## Architecture

### Convolutional Neural Network (CNN)
The MNIST system now uses a CNN architecture optimized for image recognition:

- **Layer 1 - ZeroPad2d**: Pads 28×28 images to 32×32 to preserve edge features
- **Layer 2 - Conv2d**: 16 filters with 5×5 kernel, stride 1
  - Learns basic visual patterns like edges and curves
  - Output: 28×28×16 feature maps
- **Layer 3 - BatchNorm2d**: Normalizes 16 feature channels
  - Stabilizes training and improves convergence
- **Layer 4 - ReLU**: Non-linear activation function
  - Introduces non-linearity: max(0, x)
- **Layer 5 - MaxPool2d**: 2×2 kernel with stride 2
  - Downsamples feature maps to 14×14×16
  - Reduces spatial dimensions while preserving important features
- **Layer 6 - Flatten**: Converts 3D feature maps to 1D vector
  - Flattens 16×14×14 = 3,136 features
- **Layer 7 - Linear**: Fully connected layer from 3,136 to 10 outputs
  - Produces final digit class predictions
- **Layer 8 - Softmax**: Converts logits to probabilities
  - Outputs probability distribution over 10 digit classes (0-9)

**Total Parameters**: 31,818
- Conv2d weights: 16 × 1 × 5 × 5 = 400
- Conv2d biases: 16
- BatchNorm2d: 32 (16 gamma + 16 beta)
- Linear weights: 10 × 3,136 = 31,360
- Linear biases: 10

### Performance
- **Architecture**: Convolutional Neural Network (CNN)
- **Training**: Backpropagation with gradient descent
- **Learning Rate**: 0.01
- **Training Time**: ~5-10 minutes per epoch (CPU)

## Dataset

The system uses two parquet files:
- `mnist-train.parquet`: 6,000 training samples
- `mnist-test.parquet`: 1,000 test samples

Each sample contains:
- 784 pixel values (normalized to range [0, 1])
- 1 label (digit 0-9)

## Training the Model

To train the MNIST CNN model from scratch:

```bash
# Train the model (note: CNN training is computationally expensive)
cargo run --bin mnist

# This will:
# 1. Load data from mnist-train.parquet and mnist-test.parquet
# 2. Train the CNN for 10 epochs
# 3. Display accuracy metrics during training
# 4. Show example predictions
# 5. Export the trained model to model_mnist_cnn.json

# Note: CNN training on CPU may take significant time.
# Consider using release mode for faster training:
cargo run --release --bin mnist
```

## Converting to ONNX

After training, convert the CNN model to ONNX format for browser use:

```bash
# Convert JSON model to ONNX
python3 scripts/cnn_to_onnx.py

# This creates model_mnist_cnn.onnx from model_mnist_cnn.json
```

The ONNX model includes all CNN layers:
- Pad layer (zero padding)
- Conv2d layer with learned filters
- BatchNormalization layer with running statistics
- ReLU activation
- MaxPool layer
- Flatten operation
- Gemm (fully connected) layer
- Softmax activation

This ONNX model can be used with ONNX Runtime Web for browser-based inference.

## Web Application

The web interface is located at `web/mnist.html`.

### Running the Web Application

```bash
# Start the web server
python3 scripts/start_webserver.py

# Then navigate to:
# http://localhost:8000/web/mnist.html
```

### Using the Application

1. **Click "Start Camera"** to enable your webcam
2. **Write a digit (0-9)** on white paper with a dark marker
3. **Hold the paper up** to the camera
4. **Watch the prediction** appear in real-time in the overlay

The interface shows:
- **Large digit display**: The predicted digit
- **Confidence percentage**: How confident the model is
- **Probability bars**: Confidence scores for all 10 digits
- **Inference speed**: Time taken to process each frame

### Tips for Best Results

- Use a **dark marker** on **white paper** for maximum contrast
- Make digits **large and clear**
- Ensure good **lighting**
- Hold the paper **steady** for stable predictions

## Testing the Model

You can test the trained model without a webcam:

```bash
# Test model on synthetic digit images
python3 scripts/test_mnist_model.py

# This will:
# 1. Load the trained model
# 2. Test on 100 digit images
# 3. Show per-digit accuracy
```

## File Structure

```
FraudNet/
├── mnist-train.parquet          # Training dataset (6,000 samples)
├── mnist-test.parquet           # Test dataset (1,000 samples)
├── model_mnist_cnn.json         # Trained CNN model (JSON format)
├── model_mnist_cnn.onnx         # Trained CNN model (ONNX format)
├── src/
│   ├── cnn.rs                   # CNN implementation (Conv2d, BatchNorm2d, MaxPool2d)
│   ├── model_export.rs          # Model serialization (includes CNN export)
│   ├── bin/
│   │   └── mnist.rs             # CNN training implementation
│   └── tests.rs                 # Unit tests including CNN tests
├── scripts/
│   └── cnn_to_onnx.py           # CNN to ONNX conversion script
└── MNIST.md                     # This documentation
```

## Implementation Details

### CNN Architecture

The CNN is implemented in `src/cnn.rs` with the following key components:

1. **Tensor4D**: 4D tensor representation for CNN operations (batch, channels, height, width)
2. **Conv2d**: Convolutional layer with configurable filters, kernel size, stride, and padding
3. **BatchNorm2d**: Batch normalization for stabilizing training
4. **MaxPool2d**: Max pooling for downsampling feature maps
5. **MnistCNN**: Complete CNN model combining all layers

### Data Loading

The MNIST training code (`src/bin/mnist.rs`) uses the `parquet` crate to load data:

```rust
// Load parquet file
let file = File::open("mnist-train.parquet")?;
let reader = SerializedFileReader::new(file)?;

// Iterate through rows
for record in reader.get_row_iter(None)? {
    // Extract 784 pixel values
    // Extract label
    // Create one-hot encoded target
}
```

### Training

Training uses backpropagation with:
- **Learning rate**: 0.01
- **Epochs**: 10
- **Batch size**: 1 (online learning)
- **Activation**: ReLU for hidden layers, Softmax for output
- **Loss function**: Cross-entropy loss
- **Optimizer**: Gradient descent (simplified - only FC layer is updated)

**Note**: The current implementation trains only the fully connected layer to keep training time reasonable. Full CNN backpropagation through convolutional layers is not yet implemented but can be added for improved accuracy.

### Web Inference

The web application:
1. **Captures video frames** at ~30 FPS from the webcam
2. **Resizes** each frame to 28×28 pixels
3. **Converts to grayscale** and normalizes to [0, 1]
4. **Runs inference** using ONNX Runtime Web
5. **Displays results** with real-time probability visualization

### Browser Compatibility

The web application requires:
- Modern browser with WebRTC support (Chrome, Firefox, Safari, Edge)
- HTTPS or localhost (for camera access)
- JavaScript enabled
- ONNX Runtime Web library (loaded from CDN)

## Performance Metrics

### Training Performance
```
Epoch 0: Loss = 0.067799, Train Acc = 85.35%, Test Acc = 83.80%
Epoch 1: Loss = 0.020232, Train Acc = 95.53%, Test Acc = 95.10%
Epoch 2: Loss = 0.010031, Train Acc = 96.88%, Test Acc = 96.10%
...
Epoch 9: Loss = 0.002389, Train Acc = 98.83%, Test Acc = 98.70%
```

### Example Predictions
```
Sample 1: Predicted = 1, True = 1, Confidence = 99.81%
Sample 2: Predicted = 9, True = 9, Confidence = 90.47%
Sample 3: Predicted = 0, True = 0, Confidence = 99.96%
Sample 4: Predicted = 4, True = 4, Confidence = 99.87%
Sample 5: Predicted = 8, True = 8, Confidence = 99.82%
```

## Future Enhancements

Potential improvements:
- [ ] Support for multi-digit recognition
- [ ] Drawing canvas for touchscreen devices
- [ ] Model export to other formats (TensorFlow.js, WebNN)
- [ ] Data augmentation during training
- [ ] Ensemble models for higher accuracy
- [ ] Mobile-optimized interface

## Troubleshooting

### Model not loading in browser
- Ensure the ONNX Runtime CDN is accessible
- Check browser console for errors
- Verify `model_mnist.onnx` exists in the project root

### Low accuracy predictions
- Ensure good lighting conditions
- Use high contrast (dark on white)
- Make digits large and clear
- Check that the camera is in focus

### Camera not accessible
- Grant camera permissions in browser
- Use HTTPS or localhost
- Check that no other application is using the camera

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
- Neural Network Architecture: Custom implementation in FraudNet
