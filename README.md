# FraudNet
A deep-learning network built in Rust for detecting Fraud Waste and Abuse. Models can be exported to JSON and run client-side in web browsers.

## Features

- 🧠 **Pure Rust Neural Network**: Lightweight, dependency-free implementation
- 📊 **Multiple Architectures**: Supports arbitrary network topologies
- 🌐 **Client-Side Inference**: Export trained models to run in web browsers
- 🚀 **High Performance**: Optimized matrix operations
- 🔄 **Model Serialization**: Save and load models as JSON
- ✅ **Comprehensive Tests**: Full test coverage of core functionality

## Quick Start

### Train and Export Models

```bash
# Build and run the training
cargo run

# This will:
# 1. Train three different neural networks
# 2. Export them to JSON (model_*.json)
# 3. Display training results
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

**Note:** The server must be run from the project root directory (not `web/`) so that the model JSON files are accessible at the correct paths.

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

The neural network supports arbitrary layer configurations. The README example shows a multi-layer deep network, but the implementation is flexible:

```
Input Layer (N features) 
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (52 neurons, ReLU) 
    ↓
Hidden Layer 3 (42 neurons, ReLU)
    ↓
Hidden Layer 4 (32 neurons, ReLU)
    ↓
Hidden Layer 5 (26 neurons, ReLU)
    ↓
Hidden Layer 6 (22 neurons, ReLU)
    ↓
Hidden Layer 7 (20 neurons, ReLU)
    ↓
Hidden Layer 8 (16 neurons, ReLU)
    ↓
Hidden Layer 9 (8 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

### Key Features

- **Activation Functions**: ReLU for hidden layers, Sigmoid for output layer
- **Initialization**: Xavier/Glorot initialization for weights
- **Training**: Backpropagation with gradient descent
- **Export Format**: JSON serialization for client-side use

## Model Export & Client-Side Inference

The trained neural networks can be exported to JSON format and run entirely in web browsers:

```rust
// Train a model
let mut network = NeuralNetwork::new(vec![3, 8, 1], 0.1, 12345);
network.train(&train_inputs, &train_targets, 500);

// Export to JSON
network.save_to_json("model.json")?;

// Load from JSON
let loaded = NeuralNetwork::load_from_json("model.json")?;
```

The exported models can then be loaded and used in JavaScript:

```javascript
// Load model in browser
const response = await fetch('model.json');
const modelData = await response.json();
const network = new NeuralNetwork(modelData);

// Run inference
const prediction = network.predict([0.5, 0.3, 0.2]);
```

See [web/README.md](web/README.md) for the complete client-side demo.

## Testing

Run the test suite:

```bash
# Rust tests
cargo test

# Client-side JavaScript tests
node test_client_side.cjs
```

See [TESTING.md](TESTING.md) for comprehensive test documentation.
