#!/usr/bin/env python3
"""
Convert FraudNet JSON models to ONNX format.

This script reads the JSON model files exported from the Rust neural network
and converts them to ONNX format that can be used with onnxruntime-web.
"""

import json
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto
from pathlib import Path


def json_to_onnx(json_path, onnx_path):
    """
    Convert a FraudNet JSON model to ONNX format.
    
    Args:
        json_path: Path to the JSON model file
        onnx_path: Path where the ONNX model will be saved
    """
    # Load JSON model
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    layer_sizes = model_data['layer_sizes']
    weights = model_data['weights']
    biases = model_data['biases']
    
    print(f"Converting {json_path} to ONNX...")
    print(f"  Architecture: {' → '.join(map(str, layer_sizes))}")
    print(f"  Layers: {len(weights)}")
    
    # Create ONNX nodes and initializers
    nodes = []
    initializers = []
    
    # Process each layer
    for i in range(len(weights)):
        is_last_layer = (i == len(weights) - 1)
        
        # Convert weight matrix to numpy array
        weight_data = np.array(weights[i]['data'], dtype=np.float32)
        weight_shape = (weights[i]['rows'], weights[i]['cols'])
        weight_tensor = weight_data.reshape(weight_shape)
        
        # Convert bias vector to numpy array
        bias_data = np.array(biases[i]['data'], dtype=np.float32)
        bias_shape = (biases[i]['rows'],)
        bias_tensor = bias_data.reshape(bias_shape)
        
        # Create weight initializer
        weight_name = f"weight_{i}"
        weight_init = helper.make_tensor(
            name=weight_name,
            data_type=TensorProto.FLOAT,
            dims=weight_shape,
            vals=weight_tensor.flatten().tolist()
        )
        initializers.append(weight_init)
        
        # Create bias initializer
        bias_name = f"bias_{i}"
        bias_init = helper.make_tensor(
            name=bias_name,
            data_type=TensorProto.FLOAT,
            dims=bias_shape,
            vals=bias_tensor.flatten().tolist()
        )
        initializers.append(bias_init)
        
        # Define layer input/output names
        if i == 0:
            input_name = "input"
        else:
            input_name = f"activation_{i-1}"
        
        matmul_output = f"matmul_{i}"
        add_output = f"add_{i}"
        
        if is_last_layer:
            activation_output = "output"
        else:
            activation_output = f"activation_{i}"
        
        # MatMul node: output = input × weight^T
        # Note: ONNX MatMul expects (batch, in_features) × (in_features, out_features)
        # But our weights are stored as (out_features, in_features)
        # So we need to transpose the weight matrix
        matmul_node = helper.make_node(
            'Gemm',  # Use Gemm for matrix multiplication with optional transpose
            inputs=[input_name, weight_name, bias_name],
            outputs=[add_output],
            name=f"Gemm_{i}",
            alpha=1.0,
            beta=1.0,
            transA=0,  # Don't transpose input
            transB=1   # Transpose weight matrix
        )
        nodes.append(matmul_node)
        
        # Activation function
        if is_last_layer:
            # Sigmoid for output layer
            activation_node = helper.make_node(
                'Sigmoid',
                inputs=[add_output],
                outputs=[activation_output],
                name=f"Sigmoid_{i}"
            )
        else:
            # ReLU for hidden layers
            activation_node = helper.make_node(
                'Relu',
                inputs=[add_output],
                outputs=[activation_output],
                name=f"Relu_{i}"
            )
        nodes.append(activation_node)
    
    # Create graph input
    graph_input = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        [1, layer_sizes[0]]  # Batch size of 1, input features
    )
    
    # Create graph output
    graph_output = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        [1, layer_sizes[-1]]  # Batch size of 1, output features
    )
    
    # Create the graph
    graph_def = helper.make_graph(
        nodes,
        'fraudnet',
        [graph_input],
        [graph_output],
        initializers
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='FraudNet')
    model_def.ir_version = 8  # Use IR version 8 for broader compatibility
    model_def.opset_import[0].version = 14
    
    # Check the model
    onnx.checker.check_model(model_def)
    
    # Save the model
    onnx.save(model_def, onnx_path)
    print(f"  ✓ Saved to {onnx_path}")
    
    return True


def main():
    """Convert all JSON models to ONNX format."""
    models = [
        ('model_linear.json', 'model_linear.onnx'),
        ('model_xor.json', 'model_xor.onnx'),
        ('model_circular.json', 'model_circular.onnx'),
    ]
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    success_count = 0
    for json_file, onnx_file in models:
        json_path = project_dir / json_file
        onnx_path = project_dir / onnx_file
        
        if not json_path.exists():
            print(f"⚠ Warning: {json_path} not found, skipping...")
            continue
        
        try:
            json_to_onnx(json_path, onnx_path)
            success_count += 1
        except Exception as e:
            print(f"✗ Error converting {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"✓ Successfully converted {success_count}/{len(models)} models to ONNX")
    print(f"{'='*50}\n")
    
    return success_count == len(models)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
