#!/usr/bin/env python3
"""
Convert FraudNet CNN JSON models to ONNX format.

This script reads the CNN JSON model files exported from the Rust implementation
and converts them to ONNX format that can be used with onnxruntime-web.
"""

import json
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from pathlib import Path


def cnn_to_onnx(json_path, onnx_path):
    """
    Convert a FraudNet CNN JSON model to ONNX format.
    
    Args:
        json_path: Path to the JSON model file
        onnx_path: Path where the ONNX model will be saved
    """
    # Load JSON model
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    print(f"Converting CNN {json_path} to ONNX...")
    print(f"  Conv2d: {model_data['conv_in_channels']} → {model_data['conv_out_channels']} channels, {model_data['conv_kernel_size']}x{model_data['conv_kernel_size']} kernel")
    print(f"  BatchNorm2d: {model_data['bn_num_features']} features")
    print(f"  MaxPool2d: {model_data['pool_kernel_size']}x{model_data['pool_kernel_size']} kernel")
    print(f"  Linear: {model_data['fc_weights']['cols']} → {model_data['fc_weights']['rows']}")
    
    nodes = []
    initializers = []
    
    # Input: [1, 1, 28, 28] (batch, channels, height, width)
    input_name = "input"
    
    # ===== 1. ZeroPad2d: 28x28 → 32x32 =====
    pad_node = helper.make_node(
        'Pad',
        inputs=[input_name, 'pads'],
        outputs=['padded'],
        mode='constant'
    )
    nodes.append(pad_node)
    
    # Pads: [top, left, bottom, right] for 2D, but ONNX uses [dim0_begin, dim1_begin, ..., dim0_end, dim1_end, ...]
    # For [1, 1, 28, 28], we pad only H and W dimensions: [0, 0, 2, 2, 0, 0, 2, 2]
    pads = np.array([0, 0, 2, 2, 0, 0, 2, 2], dtype=np.int64)
    pads_init = numpy_helper.from_array(pads, name='pads')
    initializers.append(pads_init)
    
    # ===== 2. Conv2d: 32x32x1 → 28x28x16 =====
    # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
    conv_weights = np.array(model_data['conv_weights'], dtype=np.float32)
    out_ch = model_data['conv_out_channels']
    in_ch = model_data['conv_in_channels']
    k = model_data['conv_kernel_size']
    conv_weights = conv_weights.reshape(out_ch, in_ch, k, k)
    
    conv_bias = np.array(model_data['conv_bias'], dtype=np.float32)
    
    conv_weight_init = numpy_helper.from_array(conv_weights, name='conv_weight')
    conv_bias_init = numpy_helper.from_array(conv_bias, name='conv_bias')
    initializers.extend([conv_weight_init, conv_bias_init])
    
    conv_node = helper.make_node(
        'Conv',
        inputs=['padded', 'conv_weight', 'conv_bias'],
        outputs=['conv_out'],
        kernel_shape=[k, k],
        strides=[model_data['conv_stride'], model_data['conv_stride']],
        pads=[0, 0, 0, 0]  # No padding in Conv since we already padded
    )
    nodes.append(conv_node)
    
    # ===== 3. BatchNorm2d =====
    bn_gamma = np.array(model_data['bn_gamma'], dtype=np.float32)
    bn_beta = np.array(model_data['bn_beta'], dtype=np.float32)
    bn_mean = np.array(model_data['bn_running_mean'], dtype=np.float32)
    bn_var = np.array(model_data['bn_running_var'], dtype=np.float32)
    bn_epsilon = model_data['bn_epsilon']
    
    bn_scale_init = numpy_helper.from_array(bn_gamma, name='bn_scale')
    bn_bias_init = numpy_helper.from_array(bn_beta, name='bn_bias')
    bn_mean_init = numpy_helper.from_array(bn_mean, name='bn_mean')
    bn_var_init = numpy_helper.from_array(bn_var, name='bn_var')
    initializers.extend([bn_scale_init, bn_bias_init, bn_mean_init, bn_var_init])
    
    bn_node = helper.make_node(
        'BatchNormalization',
        inputs=['conv_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
        outputs=['bn_out'],
        epsilon=bn_epsilon
    )
    nodes.append(bn_node)
    
    # ===== 4. ReLU =====
    relu_node = helper.make_node(
        'Relu',
        inputs=['bn_out'],
        outputs=['relu_out']
    )
    nodes.append(relu_node)
    
    # ===== 5. MaxPool2d: 28x28x16 → 14x14x16 =====
    pool_k = model_data['pool_kernel_size']
    pool_s = model_data['pool_stride']
    
    pool_node = helper.make_node(
        'MaxPool',
        inputs=['relu_out'],
        outputs=['pool_out'],
        kernel_shape=[pool_k, pool_k],
        strides=[pool_s, pool_s]
    )
    nodes.append(pool_node)
    
    # ===== 6. Flatten: [1, 16, 14, 14] → [1, 3136] =====
    flatten_node = helper.make_node(
        'Flatten',
        inputs=['pool_out'],
        outputs=['flattened'],
        axis=1  # Flatten from axis 1 onwards
    )
    nodes.append(flatten_node)
    
    # ===== 7. Linear: 3136 → 10 =====
    # Weight shape in JSON: (rows=10, cols=3136)
    # ONNX Gemm expects: input [1, 3136] × weight [3136, 10] = output [1, 10]
    # We need to transpose the weight
    fc_weights_data = np.array(model_data['fc_weights']['data'], dtype=np.float32)
    fc_weights = fc_weights_data.reshape(model_data['fc_weights']['rows'], model_data['fc_weights']['cols'])
    fc_weights = fc_weights.T  # Transpose to [3136, 10]
    
    fc_bias_data = np.array(model_data['fc_bias']['data'], dtype=np.float32)
    fc_bias = fc_bias_data.reshape(-1)
    
    fc_weight_init = numpy_helper.from_array(fc_weights, name='fc_weight')
    fc_bias_init = numpy_helper.from_array(fc_bias, name='fc_bias')
    initializers.extend([fc_weight_init, fc_bias_init])
    
    gemm_node = helper.make_node(
        'Gemm',
        inputs=['flattened', 'fc_weight', 'fc_bias'],
        outputs=['fc_out'],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0  # Already transposed
    )
    nodes.append(gemm_node)
    
    # ===== 8. Softmax =====
    softmax_node = helper.make_node(
        'Softmax',
        inputs=['fc_out'],
        outputs=['output'],
        axis=1
    )
    nodes.append(softmax_node)
    
    # Create graph input and output
    graph_input = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        [1, 1, 28, 28]  # [batch, channels, height, width]
    )
    
    graph_output = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        [1, 10]  # [batch, num_classes]
    )
    
    # Create the graph
    graph_def = helper.make_graph(
        nodes,
        'mnist_cnn',
        [graph_input],
        [graph_output],
        initializers
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='FraudNet-CNN')
    model_def.ir_version = 8
    model_def.opset_import[0].version = 14
    
    # Check the model
    onnx.checker.check_model(model_def)
    
    # Save the model
    onnx.save(model_def, onnx_path)
    print(f"  ✓ Saved to {onnx_path}")
    print(f"\nONNX Model Summary:")
    print(f"  Input:  [1, 1, 28, 28] (batch, channels, height, width)")
    print(f"  Output: [1, 10] (batch, num_classes)")
    print(f"  Total layers: 8 (Pad, Conv, BatchNorm, ReLU, MaxPool, Flatten, Gemm, Softmax)")
    
    return True


def main():
    """Convert CNN JSON model to ONNX format."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    json_path = project_dir / 'model_mnist_cnn.json'
    onnx_path = project_dir / 'model_mnist_cnn.onnx'
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        print("Please run: cargo run --bin mnist")
        print("to train the CNN and generate the JSON model file.")
        return False
    
    try:
        cnn_to_onnx(json_path, onnx_path)
        print(f"\n{'='*50}")
        print(f"✓ Successfully converted CNN model to ONNX")
        print(f"{'='*50}\n")
        return True
    except Exception as e:
        print(f"✗ Error converting CNN model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
