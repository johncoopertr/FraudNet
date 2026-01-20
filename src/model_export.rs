use crate::matrix::Matrix;
use crate::network::NeuralNetwork;
use crate::cnn::MnistCNN;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

/// Serializable representation of the neural network
#[derive(Serialize, Deserialize)]
pub struct SerializedNetwork {
    pub layer_sizes: Vec<usize>,
    pub learning_rate: f64,
    pub weights: Vec<SerializedMatrix>,
    pub biases: Vec<SerializedMatrix>,
}

/// Serializable representation of a CNN
#[derive(Serialize, Deserialize)]
pub struct SerializedCNN {
    pub learning_rate: f64,
    // Conv2d layer
    pub conv_in_channels: usize,
    pub conv_out_channels: usize,
    pub conv_kernel_size: usize,
    pub conv_stride: usize,
    pub conv_padding: usize,
    pub conv_weights: Vec<f64>,
    pub conv_bias: Vec<f64>,
    // BatchNorm2d layer
    pub bn_num_features: usize,
    pub bn_gamma: Vec<f64>,
    pub bn_beta: Vec<f64>,
    pub bn_running_mean: Vec<f64>,
    pub bn_running_var: Vec<f64>,
    pub bn_epsilon: f64,
    // MaxPool2d layer
    pub pool_kernel_size: usize,
    pub pool_stride: usize,
    // Fully connected layer
    pub fc_weights: SerializedMatrix,
    pub fc_bias: SerializedMatrix,
}

/// Serializable representation of a matrix
#[derive(Serialize, Deserialize)]
pub struct SerializedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl From<&Matrix> for SerializedMatrix {
    fn from(matrix: &Matrix) -> Self {
        SerializedMatrix {
            rows: matrix.rows,
            cols: matrix.cols,
            data: matrix.data.clone(),
        }
    }
}

impl From<&SerializedMatrix> for Matrix {
    fn from(sm: &SerializedMatrix) -> Self {
        Matrix {
            rows: sm.rows,
            cols: sm.cols,
            data: sm.data.clone(),
        }
    }
}

impl NeuralNetwork {
    /// Export the neural network to a JSON file
    pub fn save_to_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = SerializedNetwork {
            layer_sizes: self.layer_sizes.clone(),
            learning_rate: self.learning_rate,
            weights: self.weights.iter().map(|w| w.into()).collect(),
            biases: self.biases.iter().map(|b| b.into()).collect(),
        };
        
        let json = serde_json::to_string_pretty(&serialized)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        
        Ok(())
    }
    
    /// Export the neural network to a JSON string
    pub fn to_json_str(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = SerializedNetwork {
            layer_sizes: self.layer_sizes.clone(),
            learning_rate: self.learning_rate,
            weights: self.weights.iter().map(|w| w.into()).collect(),
            biases: self.biases.iter().map(|b| b.into()).collect(),
        };
        
        Ok(serde_json::to_string_pretty(&serialized)?)
    }
    
    /// Load a neural network from a JSON file
    pub fn load_from_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let serialized: SerializedNetwork = serde_json::from_str(&json)?;
        
        Ok(NeuralNetwork {
            weights: serialized.weights.iter().map(|w| w.into()).collect(),
            biases: serialized.biases.iter().map(|b| b.into()).collect(),
            layer_sizes: serialized.layer_sizes,
            learning_rate: serialized.learning_rate,
        })
    }
    
    /// Load a neural network from a JSON string
    pub fn from_json_str(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let serialized: SerializedNetwork = serde_json::from_str(json)?;
        
        Ok(NeuralNetwork {
            weights: serialized.weights.iter().map(|w| w.into()).collect(),
            biases: serialized.biases.iter().map(|b| b.into()).collect(),
            layer_sizes: serialized.layer_sizes,
            learning_rate: serialized.learning_rate,
        })
    }
}

impl MnistCNN {
    /// Export the CNN to a JSON file
    pub fn save_to_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = SerializedCNN {
            learning_rate: self.learning_rate,
            // Conv2d layer
            conv_in_channels: self.conv1.in_channels,
            conv_out_channels: self.conv1.out_channels,
            conv_kernel_size: self.conv1.kernel_size,
            conv_stride: self.conv1.stride,
            conv_padding: self.conv1.padding,
            conv_weights: self.conv1.weights.clone(),
            conv_bias: self.conv1.bias.clone(),
            // BatchNorm2d layer
            bn_num_features: self.bn1.num_features,
            bn_gamma: self.bn1.gamma.clone(),
            bn_beta: self.bn1.beta.clone(),
            bn_running_mean: self.bn1.running_mean.clone(),
            bn_running_var: self.bn1.running_var.clone(),
            bn_epsilon: self.bn1.epsilon,
            // MaxPool2d layer
            pool_kernel_size: self.pool.kernel_size,
            pool_stride: self.pool.stride,
            // Fully connected layer
            fc_weights: (&self.fc_weights).into(),
            fc_bias: (&self.fc_bias).into(),
        };
        
        let json = serde_json::to_string_pretty(&serialized)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        
        Ok(())
    }
    
    /// Export the CNN to a JSON string
    pub fn to_json_str(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = SerializedCNN {
            learning_rate: self.learning_rate,
            // Conv2d layer
            conv_in_channels: self.conv1.in_channels,
            conv_out_channels: self.conv1.out_channels,
            conv_kernel_size: self.conv1.kernel_size,
            conv_stride: self.conv1.stride,
            conv_padding: self.conv1.padding,
            conv_weights: self.conv1.weights.clone(),
            conv_bias: self.conv1.bias.clone(),
            // BatchNorm2d layer
            bn_num_features: self.bn1.num_features,
            bn_gamma: self.bn1.gamma.clone(),
            bn_beta: self.bn1.beta.clone(),
            bn_running_mean: self.bn1.running_mean.clone(),
            bn_running_var: self.bn1.running_var.clone(),
            bn_epsilon: self.bn1.epsilon,
            // MaxPool2d layer
            pool_kernel_size: self.pool.kernel_size,
            pool_stride: self.pool.stride,
            // Fully connected layer
            fc_weights: (&self.fc_weights).into(),
            fc_bias: (&self.fc_bias).into(),
        };
        
        Ok(serde_json::to_string_pretty(&serialized)?)
    }
}
