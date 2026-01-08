use crate::matrix::Matrix;
use crate::utils::SimpleRng;

/// Generate synthetic training and testing data for binary classification
pub struct SyntheticDataGenerator {
    seed: u64,
}

/// Generate synthetic unemployment insurance fraud detection data
pub struct FraudDataGenerator {
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

impl FraudDataGenerator {
    /// Create a new fraud data generator
    pub fn new(seed: u64) -> Self {
        FraudDataGenerator { seed }
    }

    /// Generate synthetic unemployment insurance fraud data
    /// Features (15 total):
    /// 1-3: Temporal (days since last claim, claim frequency, time anomaly)
    /// 4-7: Identity (SSN reuse, age verification, address changes, IP reuse)
    /// 8-11: Employment (employer verification, duration, wage consistency, separation risk)
    /// 12-13: Geographic (IP-address match, multi-state filing)
    /// 14-15: Behavioral (document quality, response pattern)
    ///
    /// Returns (inputs, targets) where 0 = legitimate, 1 = fraudulent
    pub fn generate_fraud_data(
        &mut self,
        n_samples: usize,
        fraud_ratio: f64,
    ) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut rng = SimpleRng::new(self.seed);
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        let n_fraud = (n_samples as f64 * fraud_ratio) as usize;
        let n_legitimate = n_samples - n_fraud;

        // Generate legitimate claims
        for _ in 0..n_legitimate {
            let features = self.generate_legitimate_claim(&mut rng);
            let input = Matrix::from_vec(15, 1, features);
            let target = Matrix::from_vec(1, 1, vec![0.0]);
            inputs.push(input);
            targets.push(target);
        }

        // Generate fraudulent claims (various types)
        for i in 0..n_fraud {
            let fraud_type = i % 5; // 5 different fraud patterns
            let features = match fraud_type {
                0 => self.generate_identity_fraud(&mut rng),
                1 => self.generate_concurrent_employment_fraud(&mut rng),
                2 => self.generate_geographic_fraud(&mut rng),
                3 => self.generate_application_fraud(&mut rng),
                _ => self.generate_mixed_fraud(&mut rng),
            };
            let input = Matrix::from_vec(15, 1, features);
            let target = Matrix::from_vec(1, 1, vec![1.0]);
            inputs.push(input);
            targets.push(target);
        }

        self.seed = self.seed.wrapping_add(n_samples as u64);
        (inputs, targets)
    }

    /// Generate features for a legitimate unemployment claim
    fn generate_legitimate_claim(&self, rng: &mut SimpleRng) -> Vec<f64> {
        vec![
            // Temporal features (1-3)
            rng.next_f64() * 0.5 + 0.3,      // Days since last claim (30-80 days normalized)
            rng.next_f64() * 0.2,            // Claim frequency (low, 0-2 claims/month)
            rng.next_f64() * 0.3,            // Time anomaly (normal business hours)
            
            // Identity features (4-7)
            rng.next_f64() * 0.1,            // SSN reuse (very low)
            rng.next_f64() * 0.2,            // Age verification (good)
            rng.next_f64() * 0.2,            // Address changes (stable)
            rng.next_f64() * 0.1,            // IP reuse (normal)
            
            // Employment features (8-11)
            rng.next_f64() * 0.2,            // Employer verification (verified)
            rng.next_f64() * 0.5 + 0.3,      // Employment duration (decent tenure)
            rng.next_f64() * 0.3,            // Wage consistency (consistent)
            rng.next_f64() * 0.3,            // Separation risk (low)
            
            // Geographic features (12-13)
            rng.next_f64() * 0.2,            // IP-address match (good)
            rng.next_f64() * 0.1,            // Multi-state filing (rare)
            
            // Behavioral features (14-15)
            rng.next_f64() * 0.3,            // Document quality (good)
            rng.next_f64() * 0.2,            // Response pattern (normal)
        ]
    }

    /// Generate features for identity fraud claim
    fn generate_identity_fraud(&self, rng: &mut SimpleRng) -> Vec<f64> {
        vec![
            // Temporal features - may be rushed or repeated
            rng.next_f64() * 0.3,            // Quick succession
            rng.next_f64() * 0.5 + 0.4,      // High frequency
            rng.next_f64() * 0.4 + 0.6,      // Unusual hours
            
            // Identity features - HIGH RISK
            rng.next_f64() * 0.4 + 0.6,      // SSN reuse (HIGH)
            rng.next_f64() * 0.5 + 0.4,      // Age verification issues
            rng.next_f64() * 0.5 + 0.4,      // Frequent address changes
            rng.next_f64() * 0.4 + 0.6,      // IP reuse (suspicious)
            
            // Employment features - mixed
            rng.next_f64() * 0.6,            // Some employer issues
            rng.next_f64() * 0.5,            // Variable duration
            rng.next_f64() * 0.6,            // Some wage inconsistency
            rng.next_f64() * 0.5,            // Medium separation risk
            
            // Geographic features
            rng.next_f64() * 0.5 + 0.3,      // IP-address mismatch
            rng.next_f64() * 0.4 + 0.4,      // Multi-state possibility
            
            // Behavioral features
            rng.next_f64() * 0.5 + 0.3,      // Document issues
            rng.next_f64() * 0.4 + 0.4,      // Pattern anomalies
        ]
    }

    /// Generate features for concurrent employment fraud
    fn generate_concurrent_employment_fraud(&self, rng: &mut SimpleRng) -> Vec<f64> {
        vec![
            // Temporal features - regular claims while working
            rng.next_f64() * 0.4 + 0.2,      // Regular timing
            rng.next_f64() * 0.4 + 0.3,      // Consistent frequency
            rng.next_f64() * 0.3,            // Normal hours (not trying to hide)
            
            // Identity features - legitimate person
            rng.next_f64() * 0.2,            // Low SSN reuse
            rng.next_f64() * 0.2,            // Good age verification
            rng.next_f64() * 0.2,            // Stable address
            rng.next_f64() * 0.2,            // Normal IP usage
            
            // Employment features - HIGH RISK
            rng.next_f64() * 0.5 + 0.5,      // Employer verification issues (hiding work)
            rng.next_f64() * 0.2,            // Short "unemployment" 
            rng.next_f64() * 0.6 + 0.4,      // Wage inconsistency (unreported income)
            rng.next_f64() * 0.5 + 0.4,      // High separation risk
            
            // Geographic features
            rng.next_f64() * 0.3,            // Geographic consistency
            rng.next_f64() * 0.2,            // Single state
            
            // Behavioral features
            rng.next_f64() * 0.4,            // Document quality okay
            rng.next_f64() * 0.5 + 0.3,      // Pattern shows evasion
        ]
    }

    /// Generate features for geographic fraud
    fn generate_geographic_fraud(&self, rng: &mut SimpleRng) -> Vec<f64> {
        vec![
            // Temporal features
            rng.next_f64() * 0.5,            // Various timing
            rng.next_f64() * 0.6 + 0.3,      // Multiple claims
            rng.next_f64() * 0.5 + 0.3,      // Unusual hours sometimes
            
            // Identity features - some issues
            rng.next_f64() * 0.5,            // Moderate SSN reuse
            rng.next_f64() * 0.4,            // Age verification okay
            rng.next_f64() * 0.6 + 0.3,      // Frequent moves
            rng.next_f64() * 0.5 + 0.3,      // Multiple IPs
            
            // Employment features
            rng.next_f64() * 0.5,            // Mixed employer verification
            rng.next_f64() * 0.4,            // Normal duration
            rng.next_f64() * 0.4,            // Wage consistency okay
            rng.next_f64() * 0.4,            // Moderate risk
            
            // Geographic features - HIGH RISK
            rng.next_f64() * 0.6 + 0.4,      // IP-address mismatch (HIGH)
            rng.next_f64() * 0.6 + 0.4,      // Multi-state filing (HIGH)
            
            // Behavioral features
            rng.next_f64() * 0.5,            // Document quality varies
            rng.next_f64() * 0.5 + 0.3,      // Suspicious patterns
        ]
    }

    /// Generate features for application fraud (duplicate/backdating)
    fn generate_application_fraud(&self, rng: &mut SimpleRng) -> Vec<f64> {
        vec![
            // Temporal features - HIGH RISK
            rng.next_f64() * 0.2,            // Very short gaps (duplicates)
            rng.next_f64() * 0.6 + 0.4,      // High frequency (HIGH)
            rng.next_f64() * 0.6 + 0.4,      // Off hours (trying to hide)
            
            // Identity features - legitimate person usually
            rng.next_f64() * 0.3,            // Some SSN reuse
            rng.next_f64() * 0.2,            // Age okay
            rng.next_f64() * 0.3,            // Address stable
            rng.next_f64() * 0.4 + 0.3,      // Same IP for duplicates
            
            // Employment features - fabricated
            rng.next_f64() * 0.5 + 0.4,      // Employer verification issues
            rng.next_f64() * 0.5,            // Variable duration
            rng.next_f64() * 0.5 + 0.3,      // Wage inconsistency
            rng.next_f64() * 0.5 + 0.4,      // High separation risk
            
            // Geographic features
            rng.next_f64() * 0.4,            // Some mismatch
            rng.next_f64() * 0.3,            // Possible multi-state
            
            // Behavioral features - HIGH RISK
            rng.next_f64() * 0.5 + 0.4,      // Document quality issues
            rng.next_f64() * 0.6 + 0.4,      // Very suspicious patterns
        ]
    }

    /// Generate features for mixed fraud (multiple fraud types)
    fn generate_mixed_fraud(&self, rng: &mut SimpleRng) -> Vec<f64> {
        vec![
            // Temporal features - suspicious
            rng.next_f64() * 0.4,            // Quick timing
            rng.next_f64() * 0.5 + 0.4,      // High frequency
            rng.next_f64() * 0.5 + 0.4,      // Unusual hours
            
            // Identity features - multiple red flags
            rng.next_f64() * 0.4 + 0.5,      // High SSN reuse
            rng.next_f64() * 0.5 + 0.4,      // Age verification issues
            rng.next_f64() * 0.5 + 0.4,      // Address instability
            rng.next_f64() * 0.5 + 0.4,      // IP reuse issues
            
            // Employment features - fabricated
            rng.next_f64() * 0.5 + 0.4,      // Unverifiable employer
            rng.next_f64() * 0.5,            // Inconsistent duration
            rng.next_f64() * 0.5 + 0.4,      // Wage problems
            rng.next_f64() * 0.6 + 0.4,      // High separation risk
            
            // Geographic features - red flags
            rng.next_f64() * 0.5 + 0.4,      // IP-address mismatch
            rng.next_f64() * 0.5 + 0.4,      // Multi-state filing
            
            // Behavioral features - all suspicious
            rng.next_f64() * 0.5 + 0.4,      // Document quality poor
            rng.next_f64() * 0.6 + 0.4,      // Very abnormal patterns
        ]
    }
}
