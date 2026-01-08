#!/usr/bin/env python3
"""
Test ONNX models to verify they produce correct outputs.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


def test_model(model_path, test_cases):
    """Test an ONNX model with given test cases."""
    print(f"\nTesting {model_path.name}...")
    
    # Create inference session
    session = ort.InferenceSession(str(model_path))
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"  Input: {input_name}, Output: {output_name}")
    
    # Run test cases
    for i, (inputs, expected_label) in enumerate(test_cases):
        # Prepare input
        input_data = np.array([inputs], dtype=np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_data})
        prediction = outputs[0][0][0]
        
        # Check prediction
        predicted_label = 1 if prediction >= 0.5 else 0
        match = "✓" if predicted_label == expected_label else "✗"
        
        print(f"  {match} Test {i+1}: Input {inputs} -> {prediction:.4f} (expected class {expected_label}, got {predicted_label})")
    
    return True


def main():
    """Test all ONNX models."""
    project_dir = Path(__file__).parent.parent
    
    # Test linear model
    linear_model = project_dir / "model_linear.onnx"
    if linear_model.exists():
        linear_tests = [
            ([0.5, 0.3, 0.2], 1),   # Positive example
            ([-0.5, -0.3, -0.2], 0), # Negative example
        ]
        test_model(linear_model, linear_tests)
    else:
        print(f"⚠ Warning: {linear_model} not found")
    
    # Test XOR model
    xor_model = project_dir / "model_xor.onnx"
    if xor_model.exists():
        xor_tests = [
            ([0.5, 0.5], 0),    # Same sign: expect 0
            ([-0.5, -0.5], 0),  # Same sign: expect 0
            ([0.5, -0.5], 1),   # Different sign: expect 1
            ([-0.5, 0.5], 1),   # Different sign: expect 1
        ]
        test_model(xor_model, xor_tests)
    else:
        print(f"⚠ Warning: {xor_model} not found")
    
    # Test circular model
    circular_model = project_dir / "model_circular.onnx"
    if circular_model.exists():
        circular_tests = [
            ([0.2, 0.2], 1),   # Inside (distance ~0.28)
            ([0.3, 0.3], 1),   # Inside (distance ~0.42)
            ([0.6, 0.6], 0),   # Outside (distance ~0.85)
            ([-0.7, 0.7], 0),  # Outside (distance ~0.99)
        ]
        test_model(circular_model, circular_tests)
    else:
        print(f"⚠ Warning: {circular_model} not found")
    
    print("\n" + "="*50)
    print("✓ All ONNX model tests completed!")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
