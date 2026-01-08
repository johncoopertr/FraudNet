# FraudNet Web Demo

Client-side neural network inference using **ONNX Runtime Web**.

## Overview

This web demo runs trained neural networks entirely in your browser using ONNX Runtime. The models are:
- Trained in Rust using backpropagation
- Exported to ONNX format (industry standard)
- Loaded and executed client-side with WebAssembly acceleration
- No server-side processing required

## Architecture

```
Rust Training → JSON Export → ONNX Conversion → Browser Inference
    (CPU)          (disk)       (Python script)    (WASM + WebGL)
```

## Running the Demo

### Quick Start

From the project root:

```bash
cargo test-web
```

This will automatically:
1. Train models if they don't exist
2. Convert them to ONNX format
3. Start a web server
4. Open your browser to the demo

### Manual Setup

```bash
# 1. Train models (if not already done)
cargo run --bin fraudnet

# 2. Convert to ONNX format
python3 scripts/json_to_onnx.py

# 3. Start web server from project root
python3 -m http.server 8000

# 4. Open browser to http://localhost:8000/web/
```

**Important**: The server must be started from the project root directory so that `model_*.onnx` files are accessible at the correct paths.

## Models

The demo includes three pre-trained models:

### 1. Linear Classifier (model_linear.onnx)
- **Architecture**: 3 → 8 → 1
- **Problem**: Linearly separable data
- **Accuracy**: 100% on test set
- **Use case**: Simple binary classification

### 2. XOR Classifier (model_xor.onnx)
- **Architecture**: 2 → 8 → 1
- **Problem**: XOR function (non-linear)
- **Accuracy**: 97% on test set
- **Use case**: Non-linearly separable patterns

### 3. Circular Boundary (model_circular.onnx)
- **Architecture**: 2 → 16 → 8 → 1
- **Problem**: Circular decision boundary
- **Accuracy**: 98% on test set
- **Use case**: Complex geometric patterns

## Technical Details

### ONNX Runtime Web

The demo uses [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) which provides:
- WebAssembly-based inference engine
- WebGL acceleration for matrix operations
- Industry-standard ONNX model format
- Cross-browser compatibility

### Model Format

Models are stored in ONNX format with:
- **Input**: Float32 tensor [1, N] where N is number of features
- **Output**: Float32 tensor [1, 1] with probability (0-1)
- **Operations**: Gemm (matrix multiply + bias), ReLU, Sigmoid
- **IR Version**: 8 for broad compatibility

### Performance

- **Model Loading**: < 100ms for all three models
- **Inference Time**: < 1ms per prediction
- **Memory Usage**: < 1MB total
- **Network Traffic**: Zero after initial load (all local)

## How It Works

1. **Training** (Rust):
   ```rust
   let mut network = NeuralNetwork::new(vec![3, 8, 1], 0.1, 12345);
   network.train(&inputs, &targets, 500);
   network.save_to_json("model.json")?;
   ```

2. **Conversion** (Python):
   ```python
   # Convert JSON to ONNX using onnx helper
   model_def = helper.make_model(graph_def, producer_name='FraudNet')
   onnx.save(model_def, "model.onnx")
   ```

3. **Inference** (Browser):
   ```javascript
   const session = await ort.InferenceSession.create('model.onnx');
   const tensor = new ort.Tensor('float32', inputData, [1, 3]);
   const results = await session.run({ input: tensor });
   const prediction = results.output.data[0];
   ```

## Browser Compatibility

Tested and working on:
- ✅ Chrome/Edge (Chromium)
- ✅ Firefox
- ✅ Safari
- ✅ Mobile browsers (iOS Safari, Chrome Android)

Requires:
- ES2017+ JavaScript support
- WebAssembly support
- Fetch API

## Troubleshooting

### Models not loading?

**Error**: `Failed to fetch model_*.onnx`

**Solution**: Make sure the web server is running from the project root directory:

```bash
# ❌ Wrong
cd web && python3 -m http.server 8000

# ✅ Correct
python3 -m http.server 8000  # from project root
```

### ONNX Runtime errors?

**Error**: `Cannot find module 'ort'`

**Solution**: The demo uses ONNX Runtime from CDN. Make sure you have internet connectivity.

### Models not found after training?

**Solution**: Run the conversion script:

```bash
python3 scripts/json_to_onnx.py
```

## Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [ONNX Model Format Specification](https://onnx.ai/)
- [FraudNet Main README](../README.md)
- [Testing Guide](../TESTING.md)
