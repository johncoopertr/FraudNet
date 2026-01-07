use crate::matrix::Matrix;
use crate::network::NeuralNetwork;
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
