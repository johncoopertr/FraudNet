# FraudNet
A deep-learning network built in Rust for detecting Fraud, Waste, and Abuse in unemployment insurance systems, plus MNIST digit recognition for OCR. Models are trained in Rust and exported to ONNX format for client-side inference using ONNX Runtime.

## Overview

FraudNet provides two main applications:

### 1. Unemployment Insurance Fraud Detection
Specifically designed to detect unemployment insurance fraud using a deep neural network with 15 input features covering temporal patterns, identity verification, employment history, geographic consistency, and behavioral indicators. The system can identify:

- **Identity Fraud**: Stolen or synthetic identities
- **Concurrent Employment Fraud**: Working while claiming benefits
- **Geographic Fraud**: Multi-state filings and location mismatches
- **Application Fraud**: Duplicate claims and backdating
- **Mixed Fraud**: Multiple fraud patterns combined

See [FRAUD_DETECTION.md](FRAUD_DETECTION.md) for detailed documentation on fraud detection features and methodology.

A deep-learning neural network built in Rust for detecting unemployment insurance fraud. Models are trained in Rust and exported to ONNX format for client-side inference in web browsers.

### 2. MNIST Digit Recognition (NEW!)
Real-time handwritten digit recognition using webcam OCR:

- **Training Accuracy**: 98.83%
- **Test Accuracy**: 98.70%
- **Architecture**: 784→128→64→10 neural network
- **Features**: Live webcam feed, real-time predictions, probability visualization
- **Deployment**: ONNX model running entirely in the browser

See [MNIST.md](MNIST.md) for complete MNIST documentation and usage guide.

## Features

- 🧠 **Pure Rust Neural Network**: Lightweight, dependency-free training implementation
- 🎯 **Dual Purpose**: Fraud detection + MNIST digit recognition
- 📊 **Multiple Architectures**: Supports arbitrary network topologies
- 🌐 **Client-Side Inference**: Export trained models to ONNX for browser execution
- 📹 **Webcam OCR**: Real-time digit recognition from video feed
- 🚀 **High Performance**: Optimized matrix operations and WebAssembly acceleration
- 🔄 **ONNX Format**: Industry-standard model format with ONNX Runtime Web
- ✅ **Comprehensive Tests**: Full test coverage of core functionality
- 🔍 **Multiple Fraud Types**: Detects identity, employment, geographic, and application fraud
- 💾 **Real Data Support**: Load training data from Microsoft SQL Server database or Parquet files
- 🧪 **Synthetic Fallback**: Automatic fallback to synthetic data when database is unavailable
- 🧠 Pure Rust neural network implementation
- 🎯 15-feature fraud detection model for unemployment insurance claims
- 🌐 ONNX export for client-side browser inference
- 🔍 Detects identity, employment, geographic, and application fraud
- 💾 Supports real data (SQL Server) or synthetic data training
- 📊 Deep architecture: 15→64→52→42→32→26→22→20→16→8→1

## Quick Start

### Build and Run

```bash
# Build the project
cargo build

# Train models (uses synthetic data by default)
cargo run --bin fraudnet

# Run tests
cargo test

# Run web demo (trains models if needed, starts server, opens browser)
cargo test-web
```

### Convert Models to ONNX

### MNIST Digit Recognition (Quick Start)

Train the MNIST network and run the webcam OCR application:

```bash
# Train the MNIST model
cargo run --bin mnist

# Convert to ONNX format
python3 scripts/json_to_onnx.py

# Start the web server
python3 scripts/start_webserver.py

# Open http://localhost:8000/web/mnist.html
# Click "Start Camera" and point your webcam at handwritten digits!
```

The MNIST datasets (`mnist-train.parquet` and `mnist-test.parquet`) are included in the repository with 6,000 training and 1,000 test samples. The trained model achieves 98.70% test accuracy. See [MNIST.md](MNIST.md) for complete documentation.

### Train and Export Models
After training, convert JSON models to ONNX format for browser use:

```bash
python3 scripts/json_to_onnx.py
```

### Train with Real Data (Optional)

To use real data from Microsoft SQL Server:

```bash
# Configure database connection
cp .env.example .env
# Edit .env with your database credentials

# Train with database
cargo run --bin fraudnet --features database
```

See [DATABASE_SETUP.md](DATABASE_SETUP.md) for database setup details.

## Architecture

Deep neural network optimized for fraud detection:

- **Input**: 15 features (temporal, identity, employment, geographic, behavioral)
- **Hidden Layers**: 9 layers (64→52→42→32→26→22→20→16→8 neurons, ReLU activation)
- **Output**: 1 neuron (Sigmoid) - fraud probability score
- **Training**: Backpropagation with gradient descent, Xavier initialization
- **Export**: ONNX format for browser inference

**Fraud Risk Scores:**
- 0.0-0.3: Low risk (legitimate)
- 0.3-0.6: Medium risk (review)
- 0.6-1.0: High risk (likely fraud)

See [FRAUD_DETECTION.md](FRAUD_DETECTION.md) for detailed feature descriptions.

## Client-Side Inference

Models run entirely in web browsers using ONNX Runtime Web. See [web/README.md](web/README.md) for the browser demo and [ONNX_MIGRATION.md](ONNX_MIGRATION.md) for implementation details.

## Testing

```bash
# Run all tests
cargo test

# Test ONNX models
python3 scripts/test_onnx_models.py
```

See [TESTING.md](TESTING.md) for details.

## Documentation

- [FRAUD_DETECTION.md](FRAUD_DETECTION.md) - Fraud detection features and methodology
- [DATABASE_SETUP.md](DATABASE_SETUP.md) - Database configuration and schema
- [TESTING.md](TESTING.md) - Testing guide
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage instructions
- [ONNX_MIGRATION.md](ONNX_MIGRATION.md) - ONNX implementation details
- [web/README.md](web/README.md) - Browser demo documentation
