pub mod activation;
pub mod data;
pub mod matrix;
pub mod model_export;
pub mod network;
pub mod utils;
pub mod db_schema;
pub mod db_loader;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export commonly used types
pub use matrix::Matrix;
pub use network::NeuralNetwork;
pub use data::{SyntheticDataGenerator, FraudDataGenerator};
pub use db_schema::ClaimRecord;
pub use db_loader::{DatabaseConfig, load_from_database, split_data, records_to_training_data, export_demo_data};
