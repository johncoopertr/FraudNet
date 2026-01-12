# FraudNet
A deep-learning network built in Rust for detecting Fraud, Waste, and Abuse in unemployment insurance systems. Models are trained in Rust and exported to ONNX format for client-side inference using ONNX Runtime.

## Overview

FraudNet is specifically designed to detect unemployment insurance fraud using a deep neural network with 15 input features covering temporal patterns, identity verification, employment history, geographic consistency, and behavioral indicators. The system can identify:

- **Identity Fraud**: Stolen or synthetic identities
- **Concurrent Employment Fraud**: Working while claiming benefits
- **Geographic Fraud**: Multi-state filings and location mismatches
- **Application Fraud**: Duplicate claims and backdating
- **Mixed Fraud**: Multiple fraud patterns combined

See [FRAUD_DETECTION.md](FRAUD_DETECTION.md) for detailed documentation on fraud detection features and methodology.

## Features

- 🧠 **Pure Rust Neural Network**: Lightweight, dependency-free training implementation
- 🎯 **Fraud Detection Focus**: 15 features for unemployment insurance fraud detection
- 📊 **Multiple Architectures**: Supports arbitrary network topologies (primary: 15→64→52→42→32→26→22→20→16→8→1)
- 🌐 **Client-Side Inference**: Export trained models to ONNX for browser execution
- 🚀 **High Performance**: Optimized matrix operations and WebAssembly acceleration
- 🔄 **ONNX Format**: Industry-standard model format with ONNX Runtime Web
- ✅ **Comprehensive Tests**: Full test coverage of core functionality
- 🔍 **Multiple Fraud Types**: Detects identity, employment, geographic, and application fraud
- 💾 **Real Data Support**: Load training data from PostgreSQL database
- 🧪 **Synthetic Fallback**: Automatic fallback to synthetic data when database is unavailable

## Quick Start

### Train with Real Data (Optional)

FraudNet can consume real unemployment insurance claim data from a PostgreSQL database:

```bash
# 1. Copy the example environment file
cp .env.example .env

# 2. Edit .env and configure your database connection
# DATABASE_URL=postgres://username:password@host:port/database

# 3. Build and run with database support
cargo run --features database

# This will:
# - Connect to your PostgreSQL database
# - Load and validate claim records
# - Split data into training (70%), testing (20%), and demo (10%) sets
# - Train the neural network on real data
# - Export demo records to demo_data.json for web demonstration
```

See [DATABASE_SETUP.md](DATABASE_SETUP.md) for detailed database setup instructions, including the required table schema and feature descriptions.

### Train and Export Models

```bash
# Build and run the training (uses synthetic data if no database configured)
cargo run --bin fraudnet

# This will:
# 1. Train the fraud detection neural network (primary model)
# 2. Train three validation networks (linear, XOR, circular)
# 3. Export all models to JSON (model_*.json)
# 4. Display training results and example fraud predictions

# Convert JSON models to ONNX format
python3 scripts/json_to_onnx.py

# This creates model_*.onnx files for client-side use
# Primary model: model_fraud_detection.onnx
```

### Run Client-Side Demo

**Easiest way** - Use the cargo test-web command:

```bash
# Train models (if needed) and start web demo
cargo test-web

# This will:
# 1. Check if model files exist, train them if needed
# 2. Start a web server at http://localhost:8000
# 3. Automatically open the demo in your browser
```

**Manual method** - Start server from project root:

```bash
# From the project root (not the web/ directory)
python3 -m http.server 8000

# Open browser to http://localhost:8000/web/
```

**Note:** The server must be run from the project root directory (not `web/`) so that the ONNX model files are accessible at the correct paths.

See [web/README.md](web/README.md) for more details on the browser demo.

## Development Environment

This project uses Nix flakes for reproducible development environments. The flake provides:
- Rust toolchain (stable) with rust-analyzer, clippy, and rustfmt via oxalica overlay
- WASM target for web compilation
- ONNX runtime and dependencies
- Development tools (cargo-watch, cargo-edit, bacon, etc.)

### Getting Started with Nix

1. **Install Nix** with flakes enabled:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
   ```

2. **Enter the development shell**:
   ```bash
   nix develop
   ```

3. **Or use direnv** for automatic environment loading:
   ```bash
   direnv allow
   ```

### Building the Project

```bash
# Build the project
cargo build

# Run tests
cargo test

# Train models and export to JSON
cargo run
```

## Architecture

The primary fraud detection network uses a deep architecture optimized for detecting complex fraud patterns:

```
Input Layer (15 features)
    ↓ Temporal (3): Days since last claim, claim frequency, time anomaly
    ↓ Identity (4): SSN reuse, age verification, address changes, IP reuse
    ↓ Employment (4): Employer verification, duration, wage consistency, separation risk
    ↓ Geographic (2): IP-address match, multi-state filing
    ↓ Behavioral (2): Document quality, response patterns
    ↓
Hidden Layer 1 (64 neurons, ReLU) - Learn complex feature interactions
    ↓
Hidden Layer 2 (52 neurons, ReLU) - Dimensionality reduction
    ↓
Hidden Layer 3 (42 neurons, ReLU) - Pattern abstraction
    ↓
Hidden Layer 4 (32 neurons, ReLU) - Further refinement
    ↓
Hidden Layer 5 (26 neurons, ReLU) - Compressed representation
    ↓
Hidden Layer 6 (22 neurons, ReLU) - Deep feature extraction
    ↓
Hidden Layer 7 (20 neurons, ReLU) - Subtle pattern detection
    ↓
Hidden Layer 8 (16 neurons, ReLU) - Final refinement
    ↓
Hidden Layer 9 (8 neurons, ReLU) - Distilled features
    ↓
Output Layer (1 neuron, Sigmoid) - Fraud probability (0.0 to 1.0)
```

**Output Interpretation:**
- 0.0 - 0.3: Low risk (legitimate claim)
- 0.3 - 0.6: Medium risk (flag for review)
- 0.6 - 1.0: High risk (likely fraud)

### Key Features

- **Activation Functions**: ReLU for hidden layers, Sigmoid for output layer
- **Initialization**: Xavier/Glorot initialization for weights
- **Training**: Backpropagation with gradient descent
- **Export Format**: ONNX format for universal compatibility
- **Input Features**: 15 carefully selected fraud indicators
- **Fraud Types**: Detects 5 different fraud patterns (identity, employment, geographic, application, mixed)

## Model Export & Client-Side Inference

The trained fraud detection neural network is exported to ONNX format and can run entirely in web browsers using ONNX Runtime Web:

```rust
// Train the fraud detection model
let mut fraud_gen = FraudDataGenerator::new(42);
let (inputs, targets) = fraud_gen.generate_fraud_data(1000, 0.30);
let mut network = NeuralNetwork::new(vec![15, 64, 52, 42, 32, 26, 22, 20, 16, 8, 1], 0.08, 12345);
network.train(&inputs, &targets, 1500);

// Export to JSON
network.save_to_json("model_fraud_detection.json")?;
```

Then convert to ONNX:

```bash
python3 scripts/json_to_onnx.py
```

The ONNX models can be loaded in the browser using ONNX Runtime Web for real-time fraud detection:

```javascript
// Load fraud detection model in browser
const session = await ort.InferenceSession.create('model_fraud_detection.onnx');

// Prepare claim features (15 features)
const claimFeatures = [
    0.4, 0.1, 0.2,        // Temporal features
    0.05, 0.1, 0.1, 0.05, // Identity features
    0.1, 0.6, 0.2, 0.2,   // Employment features
    0.1, 0.05,             // Geographic features
    0.2, 0.1               // Behavioral features
];

// Run fraud detection inference
const inputTensor = new ort.Tensor('float32', claimFeatures, [1, 15]);
const results = await session.run({ input: inputTensor });
const fraudScore = results.output.data[0];

// Interpret result
if (fraudScore >= 0.6) {
    console.log(`High fraud risk: ${fraudScore.toFixed(4)}`);
} else if (fraudScore >= 0.3) {
    console.log(`Medium fraud risk: ${fraudScore.toFixed(4)} - flag for review`);
} else {
    console.log(`Low fraud risk: ${fraudScore.toFixed(4)}`);
}
```

See [web/README.md](web/README.md) for the complete client-side demo.

## Testing

Run the test suite:

```bash
# Rust tests
cargo test

# Test ONNX model export and inference
python3 scripts/test_onnx_models.py

# Interactive web demo (with automatic server setup)
cargo test-web
```

See [TESTING.md](TESTING.md) for comprehensive test documentation.

## Troubleshooting

### Models not loading in browser?

The most common issue is running the web server from the wrong directory. The model JSON files need to be accessible from the server root:

**❌ Wrong:**
```bash
cd web
python3 -m http.server 8000  # Models won't load!
```

**✅ Correct:**
```bash
# From project root
python3 -m http.server 8000  # Then visit http://localhost:8000/web/

# Or use the cargo command (recommended)
cargo test-web
```

### Port already in use?

If you get "Address already in use" error:
```bash
# Find and stop the process using port 8000
lsof -ti:8000 | xargs kill
```
