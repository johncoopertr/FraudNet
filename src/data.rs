use crate::matrix::Matrix;
use crate::utils::SimpleRng;

/// Generate synthetic training and testing data for binary classification
pub struct SyntheticDataGenerator {
    seed: u64,
}

impl SyntheticDataGenerator {
    pub fn new(seed: u64) -> Self {
        SyntheticDataGenerator { seed }
    }

    /// Generate linearly separable data for binary classification
    /// Returns (inputs, targets) where each input is a column vector
    pub fn generate_linear_separable(
        &mut self,
        n_samples: usize,
        n_features: usize,
    ) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut rng = SimpleRng::new(self.seed);
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for _ in 0..n_samples {
            let mut features = Vec::new();

            // Generate random features
            for _ in 0..n_features {
                features.push(rng.next_f64() * 2.0 - 1.0); // Range [-1, 1]
            }

            // Simple linear decision boundary: sum of features > 0
            let sum: f64 = features.iter().sum();
            let label = if sum > 0.0 { 1.0 } else { 0.0 };

            let input = Matrix::from_vec(n_features, 1, features);
            let target = Matrix::from_vec(1, 1, vec![label]);

            inputs.push(input);
            targets.push(target);
        }

        self.seed = self.seed.wrapping_add(n_samples as u64);
        (inputs, targets)
    }

    /// Generate XOR-like data (non-linearly separable)
    /// Works with 2 features, returns (inputs, targets)
    pub fn generate_xor(&mut self, n_samples: usize) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut rng = SimpleRng::new(self.seed);
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for _ in 0..n_samples {
            let x1 = rng.next_f64() * 2.0 - 1.0; // Range [-1, 1]
            let x2 = rng.next_f64() * 2.0 - 1.0;

            // XOR logic: different signs = 1, same signs = 0
            let label = if x1 * x2 < 0.0 { 1.0 } else { 0.0 };

            let input = Matrix::from_vec(2, 1, vec![x1, x2]);
            let target = Matrix::from_vec(1, 1, vec![label]);

            inputs.push(input);
            targets.push(target);
        }

        self.seed = self.seed.wrapping_add(n_samples as u64);
        (inputs, targets)
    }

    /// Generate circular data (non-linearly separable)
    /// Points inside circle are class 1, outside are class 0
    pub fn generate_circular(
        &mut self,
        n_samples: usize,
        radius: f64,
    ) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut rng = SimpleRng::new(self.seed);
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for _ in 0..n_samples {
            let x1 = rng.next_f64() * 2.0 - 1.0; // Range [-1, 1]
            let x2 = rng.next_f64() * 2.0 - 1.0;

            // Points inside circle get label 1, outside get 0
            let distance = (x1 * x1 + x2 * x2).sqrt();
            let label = if distance < radius { 1.0 } else { 0.0 };

            let input = Matrix::from_vec(2, 1, vec![x1, x2]);
            let target = Matrix::from_vec(1, 1, vec![label]);

            inputs.push(input);
            targets.push(target);
        }

        self.seed = self.seed.wrapping_add(n_samples as u64);
        (inputs, targets)
    }
}
