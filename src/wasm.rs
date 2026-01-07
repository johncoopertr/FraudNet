use wasm_bindgen::prelude::*;
use crate::matrix::Matrix;
use crate::network::NeuralNetwork;

#[wasm_bindgen]
pub struct WasmNeuralNetwork {
    network: NeuralNetwork,
}

#[wasm_bindgen]
impl WasmNeuralNetwork {
    /// Create a new neural network with the given layer sizes
    #[wasm_bindgen(constructor)]
    pub fn new(layer_sizes: Vec<usize>, learning_rate: f64, seed: u64) -> Self {
        console_error_panic_hook::set_once();
        WasmNeuralNetwork {
            network: NeuralNetwork::new(layer_sizes, learning_rate, seed),
        }
    }
    
    /// Load a network from JSON string
    #[wasm_bindgen]
    pub fn from_json(json: &str) -> Result<WasmNeuralNetwork, JsValue> {
        console_error_panic_hook::set_once();
        let network = NeuralNetwork::from_json_str(json)
            .map_err(|e| JsValue::from_str(&format!("Failed to load network: {}", e)))?;
        Ok(WasmNeuralNetwork { network })
    }
    
    /// Export the network to JSON string
    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.network
            .to_json_str()
            .map_err(|e| JsValue::from_str(&format!("Failed to export network: {}", e)))
    }
    
    /// Predict output for given input
    #[wasm_bindgen]
    pub fn predict(&self, input: Vec<f64>) -> Result<Vec<f64>, JsValue> {
        let input_matrix = Matrix::from_vec(input.len(), 1, input);
        let output = self.network.predict(&input_matrix);
        Ok(output.data.clone())
    }
    
    /// Train the network on a single example
    #[wasm_bindgen]
    pub fn train_step(&mut self, input: Vec<f64>, target: Vec<f64>) -> Result<f64, JsValue> {
        let input_matrix = Matrix::from_vec(input.len(), 1, input);
        let target_matrix = Matrix::from_vec(target.len(), 1, target);
        Ok(self.network.train_step(&input_matrix, &target_matrix))
    }
}

/// Initialize the WASM module (called automatically)
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}
