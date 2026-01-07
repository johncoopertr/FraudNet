# FraudNet Web Demo

This directory contains a client-side web demonstration of the FraudNet neural network running in the browser.

## Features

- **Pure JavaScript Implementation**: Neural network inference implemented in vanilla JavaScript
- **No Server Required**: All predictions run locally in your browser
- **Trained Models**: Pre-trained models exported from Rust and loaded as JSON
- **Interactive Demo**: Three different classification problems to explore

## Running the Demo

### Option 1: Cargo Command (Recommended)

```bash
# From the project root directory
cargo test-web
```

This command will:
1. Check if trained models exist, and train them if needed
2. Start a web server at http://localhost:8000
3. Automatically open the demo in your browser (if available)
4. Serve files from the project root so models are accessible

### Option 2: Manual Python Server

```bash
# From the project root directory (NOT the web/ directory)
python3 -m http.server 8000
```

Then open your browser to: http://localhost:8000/web/

**Important:** The server must be started from the project root directory, not the `web/` directory, so that the model JSON files (`model_*.json`) are accessible at the correct paths.

### Option 3: Using Node.js http-server

```bash
# Install http-server globally (if not already installed)
npm install -g http-server

# From the project root directory
http-server
```

Then navigate to: http://localhost:8080/web/

## Demo Models

### 1. Linear Classifier (3 → 8 → 1)
- **Problem**: Linearly separable data
- **Accuracy**: 100% on test set
- **Use Case**: Simple decision boundaries

### 2. XOR Classifier (2 → 8 → 1)
- **Problem**: XOR function (non-linearly separable)
- **Accuracy**: 97% on test set
- **Use Case**: Classic non-linear problem

### 3. Circular Boundary Classifier (2 → 16 → 8 → 1)
- **Problem**: Circular decision boundary
- **Accuracy**: 98% on test set
- **Use Case**: Complex geometric patterns

## Architecture

The client-side implementation includes:

- **Matrix Operations**: Pure JavaScript matrix class for computations
- **Activation Functions**: ReLU (hidden layers) and Sigmoid (output layer)
- **Forward Pass**: Complete neural network inference
- **Model Loading**: JSON deserialization of trained weights

## Files

- `index.html` - Main demo page with UI
- `fraudnet.js` - Neural network implementation in JavaScript
- `../model_*.json` - Exported trained models (in project root)

## Technical Details

### Model Format

Models are exported as JSON with the following structure:

```json
{
  "layer_sizes": [input_size, hidden1, hidden2, ..., output_size],
  "learning_rate": 0.1,
  "weights": [
    {
      "rows": output_size,
      "cols": input_size,
      "data": [...]
    },
    ...
  ],
  "biases": [
    {
      "rows": output_size,
      "cols": 1,
      "data": [...]
    },
    ...
  ]
}
```

### Inference Process

1. Load model JSON from file
2. Deserialize weights and biases into Matrix objects
3. For each prediction:
   - Forward pass through layers
   - Apply ReLU activation to hidden layers
   - Apply Sigmoid activation to output layer
   - Return prediction value

## Browser Compatibility

Tested and working on:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

Requires:
- Modern JavaScript (ES6+)
- Fetch API support
- JSON parsing

## Performance

- Model loading: < 100ms
- Single prediction: < 1ms
- No network requests after initial load
- All computation happens client-side
