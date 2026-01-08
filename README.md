# FraudNet
A deep-learning network built in Rust for detecting Fraud Waste and Abuse. Models are trained in Rust and exported to ONNX format for client-side inference using ONNX Runtime.

## Features

- 🧠 **Pure Rust Neural Network**: Lightweight, dependency-free training implementation
- 📊 **Multiple Architectures**: Supports arbitrary network topologies
- 🌐 **Client-Side Inference**: Export trained models to ONNX for browser execution
- 🚀 **High Performance**: Optimized matrix operations and WebAssembly acceleration
- 🔄 **ONNX Format**: Industry-standard model format with ONNX Runtime Web
- ✅ **Comprehensive Tests**: Full test coverage of core functionality

## Quick Start

### Train and Export Models

```bash
# Build and run the training
cargo run --bin fraudnet

# This will:
# 1. Train three different neural networks
# 2. Export them to JSON (model_*.json)
# 3. Display training results

# Convert JSON models to ONNX format
python3 scripts/json_to_onnx.py

# This creates model_*.onnx files for client-side use
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
- **Export Format**: ONNX format for universal compatibility

## Model Export & Client-Side Inference

The trained neural networks are exported to ONNX format and run entirely in web browsers using ONNX Runtime Web:

```rust
// Train a model
let mut network = NeuralNetwork::new(vec![3, 8, 1], 0.1, 12345);
network.train(&train_inputs, &train_targets, 500);

// Export to JSON
network.save_to_json("model.json")?;
```

Then convert to ONNX:

```bash
python3 scripts/json_to_onnx.py
```

The ONNX models can be loaded in the browser using ONNX Runtime Web:

```javascript
// Load model in browser
const session = await ort.InferenceSession.create('model.onnx');

// Run inference
const inputTensor = new ort.Tensor('float32', [0.5, 0.3, 0.2], [1, 3]);
const results = await session.run({ input: inputTensor });
const prediction = results.output.data[0];
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
