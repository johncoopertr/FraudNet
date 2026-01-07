# FraudNet Web Demo Guide

## Quick Start

1. **Start the web server**:
   ```bash
   cd web
   python3 -m http.server 8000
   ```

2. **Open your browser** to: http://localhost:8000

3. **Try the demos**:
   - Adjust input values using the number fields
   - Click "Predict" buttons to see results
   - Watch predictions update in real-time

## Demo Features

### Visual Interface
- **Clean, Modern Design**: Gradient background with card-based layout
- **Responsive Grid**: Automatically adjusts to screen size
- **Color-Coded Results**: 
  - Green for positive predictions
  - Red for negative predictions
- **Real-Time Feedback**: Instant results when you click predict

### Model Cards

#### 1. Linear Classifier
- **Input Fields**: 3 features (Feature 1, Feature 2, Feature 3)
- **Test Values**: Try [0.5, 0.3, 0.2] for Class 1
- **Test Values**: Try [-0.5, -0.3, -0.2] for Class 0
- **Logic**: Predicts Class 1 if sum of features > 0

#### 2. XOR Classifier
- **Input Fields**: X1 and X2 (range: -1 to 1)
- **Test Values**: Try [0.5, -0.5] for XOR = 1
- **Test Values**: Try [0.5, 0.5] for XOR = 0
- **Logic**: Predicts 1 when signs of inputs differ

#### 3. Circular Boundary Classifier
- **Input Fields**: X1 and X2 (range: -1 to 1)
- **Test Values**: Try [0.3, 0.3] for Inside
- **Test Values**: Try [0.7, 0.7] for Outside
- **Logic**: Predicts "Inside" when distance from origin < 0.5

## Interactive Features

### Input Controls
- **Number Fields**: Precise value entry
- **Step Controls**: Fine-tune values with ± buttons
- **Range Limits**: Prevents invalid inputs

### Results Display
Shows for each prediction:
- **Classification**: Human-readable result
- **Probability**: Percentage confidence
- **Raw Output**: Exact neural network output
- **Additional Info**: Model-specific details

### Status Indicator
- **Loading**: Shows when models are being loaded
- **Ready**: Green checkmark when all models loaded
- **Error**: Red indicator if loading fails

## Example Usage

### Example 1: Linear Classification
```
Input:  [0.8, 0.5, 0.3]
Sum:    1.6 (positive)
Output: Class 1 (100% confidence)
```

### Example 2: XOR Problem
```
Input:  [0.7, -0.6]
Signs:  Different (+ and -)
Output: XOR = 1 (100% confidence)
```

### Example 3: Circular Boundary
```
Input:    [0.4, 0.2]
Distance: 0.447
Output:   Inside Circle (98% confidence)
```

## Browser Compatibility

Tested and verified on:
- ✅ Chrome 120+
- ✅ Firefox 120+
- ✅ Safari 17+
- ✅ Edge 120+

## Troubleshooting

### Models not loading?
1. Check that you're serving from a web server (not file://)
2. Verify model JSON files are in parent directory
3. Check browser console for errors

### Predictions seem wrong?
1. Verify input values are in the expected range
2. Check that models loaded successfully
3. Try the example values provided

### Can't connect to server?
1. Ensure python server is running
2. Check port 8000 isn't already in use
3. Try a different port: `python3 -m http.server 8080`

## Technical Notes

- **No Backend Required**: All computation happens in browser
- **Privacy Friendly**: No data sent to any server
- **Offline Capable**: Works without internet after initial load
- **Fast**: Predictions complete in < 1ms

## Screenshots

When you open the demo, you should see:
1. Purple gradient background
2. "FraudNet" title at the top
3. Green status indicator showing "✓ Neural Network Models Loaded"
4. Three white cards in a grid layout
5. Input fields and predict buttons in each card
6. "About This Demo" section at the bottom

Try changing the values and clicking predict to see the neural networks in action!
