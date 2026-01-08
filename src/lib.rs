pub mod activation;
pub mod data;
pub mod matrix;
pub mod model_export;
pub mod network;
pub mod utils;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export commonly used types
pub use matrix::Matrix;
pub use network::NeuralNetwork;
pub use data::{SyntheticDataGenerator, FraudDataGenerator};
