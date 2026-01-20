# FraudNet

A deep-learning neural network built in Rust for detecting unemployment insurance fraud. Models are trained in Rust and exported to ONNX format for client-side inference in web browsers.

## Features

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
