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

```bash
# Start a local web server
cd web
python3 -m http.server 8000

# Open browser to http://localhost:8000
```

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
nix build
```

Implementation:

This is a arbitrarily-scalable neural network capable of creating a classification network of N-inputs, should look something like the following:

```
Input Layer (3 features after UMAP) 
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

Finally, the network, once trained, should be compiled to be compatible with the ONNX runtime, to be used client side in a web-browser.
