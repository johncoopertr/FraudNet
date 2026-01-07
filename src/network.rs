use crate::activation::{
    relu_derivative_matrix, relu_matrix, sigmoid_derivative_matrix, sigmoid_matrix,
};
use crate::matrix::Matrix;

/// A simple feedforward neural network
pub struct NeuralNetwork {
    /// Weights for each layer (input->hidden1, hidden1->hidden2, ..., last_hidden->output)
    pub weights: Vec<Matrix>,
    /// Biases for each layer
    pub biases: Vec<Matrix>,
    /// Layer sizes (including input and output)
    pub layer_sizes: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
}

impl NeuralNetwork {
    /// Create a new neural network with given layer sizes
    /// Example: vec![3, 64, 52, 42, 1] creates a network with:
    ///   - 3 inputs
    ///   - 3 hidden layers (64, 52, 42 neurons)
    ///   - 1 output
    pub fn new(layer_sizes: Vec<usize>, learning_rate: f64, seed: u64) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Must have at least input and output layers"
        );

        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut current_seed = seed;

        // Initialize weights and biases for each layer connection
        for i in 0..layer_sizes.len() - 1 {
            let rows = layer_sizes[i + 1];
            let cols = layer_sizes[i];

            // Xavier/Glorot initialization scaling
            let scale = (2.0 / (cols as f64)).sqrt();
            let w = Matrix::random(rows, cols, current_seed).scale(scale);
            weights.push(w);

            current_seed = current_seed.wrapping_add(1);
            let b = Matrix::zeros(rows, 1);
            biases.push(b);
        }

        NeuralNetwork {
            weights,
            biases,
            layer_sizes,
            learning_rate,
        }
    }

    /// Forward pass through the network
    /// Returns (activations, pre_activations) for all layers including input
    pub fn forward(&self, input: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        assert_eq!(input.rows, self.layer_sizes[0], "Input size mismatch");
        assert_eq!(input.cols, 1, "Input must be a column vector");

        let mut activations = vec![input.clone()];
        let mut pre_activations = vec![input.clone()];

        for i in 0..self.weights.len() {
            // z = W * a + b
            let z = self.weights[i].dot(&activations[i]).add(&self.biases[i]);
            pre_activations.push(z.clone());

            // Apply activation function
            let a = if i < self.weights.len() - 1 {
                // Hidden layers use ReLU
                relu_matrix(&z)
            } else {
                // Output layer uses Sigmoid
                sigmoid_matrix(&z)
            };
            activations.push(a);
        }

        (activations, pre_activations)
    }

    /// Predict output for given input
    pub fn predict(&self, input: &Matrix) -> Matrix {
        let (activations, _) = self.forward(input);
        activations.last().unwrap().clone()
    }

    /// Train the network on a single example using backpropagation
    pub fn train_step(&mut self, input: &Matrix, target: &Matrix) -> f64 {
        assert_eq!(
            target.rows,
            self.layer_sizes[self.layer_sizes.len() - 1],
            "Target size mismatch"
        );
        assert_eq!(target.cols, 1, "Target must be a column vector");

        // Forward pass
        let (activations, pre_activations) = self.forward(input);

        // Calculate output layer error
        let output = activations.last().unwrap();
        let output_error = output.sub(target);

        // Calculate loss (mean squared error)
        let loss = output_error.hadamard(&output_error).mean();

        // Backpropagation
        let mut deltas = Vec::new();

        // Output layer delta
        let output_z = &pre_activations[pre_activations.len() - 1];
        let output_delta = output_error.hadamard(&sigmoid_derivative_matrix(output_z));
        deltas.push(output_delta);

        // Hidden layers deltas (backward)
        for i in (1..self.weights.len()).rev() {
            let delta = self.weights[i]
                .transpose()
                .dot(&deltas[deltas.len() - 1])
                .hadamard(&relu_derivative_matrix(&pre_activations[i]));
            deltas.push(delta);
        }

        // Reverse deltas to match forward order
        deltas.reverse();

        // Update weights and biases
        for i in 0..self.weights.len() {
            let gradient = deltas[i].dot(&activations[i].transpose());
            let weight_update = gradient.scale(self.learning_rate);
            self.weights[i] = self.weights[i].sub(&weight_update);

            let bias_update = deltas[i].scale(self.learning_rate);
            self.biases[i] = self.biases[i].sub(&bias_update);
        }

        loss
    }

    /// Train the network on a batch of examples
    pub fn train(&mut self, inputs: &[Matrix], targets: &[Matrix], epochs: usize) {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Inputs and targets must have same length"
        );

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                let loss = self.train_step(input, target);
                total_loss += loss;
            }

            let avg_loss = total_loss / inputs.len() as f64;

            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!("Epoch {}: Average Loss = {:.6}", epoch, avg_loss);
            }
        }
    }

    /// Evaluate accuracy on a dataset
    pub fn evaluate(&self, inputs: &[Matrix], targets: &[Matrix], threshold: f64) -> f64 {
        let mut correct = 0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input);
            let pred_class = if prediction.get(0, 0) >= threshold {
                1.0
            } else {
                0.0
            };
            let true_class = target.get(0, 0);

            if (pred_class - true_class).abs() < 0.1 {
                correct += 1;
            }
        }

        correct as f64 / inputs.len() as f64
    }
}
