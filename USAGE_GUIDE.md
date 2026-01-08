# FraudNet Usage Guide

## Quick Start: Unemployment Insurance Fraud Detection

### Training the Model

```bash
# Build and train the fraud detection model
cargo run --bin fraudnet

# This will:
# - Train a deep neural network (15 inputs → 9 hidden layers → 1 output)
# - Generate synthetic fraud and legitimate claim data
# - Export the trained model to model_fraud_detection.json
# - Display example fraud predictions
```

### Input Features (15 total)

When using the fraud detection model, provide 15 features in this order:

#### Temporal Features (indices 0-2)
1. **Days Since Last Claim** (0.0-1.0): Normalized days between claims
2. **Claim Frequency** (0.0-1.0): Number of claims in past 30 days / 10
3. **Time of Day Anomaly** (0.0-1.0): 0=normal hours, 1=suspicious hours

#### Identity Verification Features (indices 3-6)
4. **SSN Reuse Score** (0.0-1.0): Number of recent claims with same SSN / 10
5. **Age Verification Score** (0.0-1.0): Age consistency issues (0=verified, 1=problematic)
6. **Address Change Frequency** (0.0-1.0): Number of address changes / 10
7. **IP Address Reuse Count** (0.0-1.0): Claims from same IP in 30 days / 10

#### Employment History Features (indices 7-10)
8. **Employer Verification Status** (0.0-1.0): 0=verified, 1=unverifiable
9. **Employment Duration** (0.0-1.0): Length of last employment (normalized)
10. **Wage Consistency Score** (0.0-1.0): Wage/job type inconsistency
11. **Separation Reason Risk** (0.0-1.0): Risk score for termination reason

#### Geographic Features (indices 11-12)
12. **IP-Address Geographic Match** (0.0-1.0): Distance between IP and address
13. **Multi-State Filing Indicator** (0.0-1.0): 0=single state, 1=multiple detected

#### Behavioral Features (indices 13-14)
14. **Document Quality Score** (0.0-1.0): Document authenticity issues
15. **Response Pattern Anomaly** (0.0-1.0): Unusual answer patterns

### Example: Using the Model in Rust

```rust
use fraudnet::{NeuralNetwork, Matrix};

// Load the trained fraud detection model
let network = NeuralNetwork::load_from_json("model_fraud_detection.json")?;

// Create a claim to evaluate (15 features)
let claim_features = vec![
    0.4,   // Days since last claim (normal)
    0.1,   // Claim frequency (low)
    0.2,   // Time anomaly (normal hours)
    0.05,  // SSN reuse (very low)
    0.1,   // Age verification (good)
    0.1,   // Address changes (stable)
    0.05,  // IP reuse (normal)
    0.1,   // Employer verification (verified)
    0.6,   // Employment duration (good tenure)
    0.2,   // Wage consistency (consistent)
    0.2,   // Separation risk (low)
    0.1,   // IP-address match (good)
    0.05,  // Multi-state filing (rare)
    0.2,   // Document quality (good)
    0.1,   // Response pattern (normal)
];

let input = Matrix::from_vec(15, 1, claim_features);
let prediction = network.predict(&input);
let fraud_score = prediction.get(0, 0);

// Interpret the result
if fraud_score >= 0.6 {
    println!("HIGH RISK: Fraud score {:.4}", fraud_score);
} else if fraud_score >= 0.3 {
    println!("MEDIUM RISK: Flag for review - score {:.4}", fraud_score);
} else {
    println!("LOW RISK: Legitimate claim - score {:.4}", fraud_score);
}
```

### Example: Using the Model in JavaScript (Browser)

After converting to ONNX format (`python3 scripts/json_to_onnx.py`):

```javascript
// Load the ONNX model
const session = await ort.InferenceSession.create('model_fraud_detection.onnx');

// Prepare claim features (same 15 features as above)
const claimFeatures = new Float32Array([
    0.4, 0.1, 0.2,           // Temporal
    0.05, 0.1, 0.1, 0.05,    // Identity
    0.1, 0.6, 0.2, 0.2,      // Employment
    0.1, 0.05,               // Geographic
    0.2, 0.1                 // Behavioral
]);

// Run inference
const inputTensor = new ort.Tensor('float32', claimFeatures, [1, 15]);
const results = await session.run({ input: inputTensor });
const fraudScore = results.output.data[0];

// Interpret result
if (fraudScore >= 0.6) {
    console.log(`HIGH RISK: ${fraudScore.toFixed(4)}`);
} else if (fraudScore >= 0.3) {
    console.log(`MEDIUM RISK: ${fraudScore.toFixed(4)}`);
} else {
    console.log(`LOW RISK: ${fraudScore.toFixed(4)}`);
}
```

## Interpreting Results

### Risk Thresholds

- **0.0 - 0.3**: Low risk - Likely legitimate claim
  - Action: Approve with standard processing
  
- **0.3 - 0.6**: Medium risk - Potential fraud indicators
  - Action: Flag for manual review
  - Assign to fraud investigator
  
- **0.6 - 1.0**: High risk - Multiple fraud indicators
  - Action: Immediate investigation required
  - Hold payment pending verification

### Common Fraud Patterns

1. **Identity Fraud** (high SSN reuse, IP mismatches, age issues)
   - Typical score: 0.7 - 0.9
   
2. **Concurrent Employment** (unverifiable employer, wage inconsistencies)
   - Typical score: 0.6 - 0.8
   
3. **Geographic Fraud** (multi-state, IP-address mismatches)
   - Typical score: 0.7 - 0.9
   
4. **Application Fraud** (high frequency, temporal anomalies)
   - Typical score: 0.6 - 0.8
   
5. **Mixed Fraud** (multiple red flags across categories)
   - Typical score: 0.8 - 1.0

## Feature Engineering Tips

### Data Normalization

All features should be normalized to [0, 1] range:

```rust
// Example: Normalize days since last claim (max expected: 90 days)
let days_since_last = 45.0;
let normalized = (days_since_last / 90.0).min(1.0);

// Example: Normalize claim frequency (max expected: 10 claims)
let num_claims = 3.0;
let normalized = (num_claims / 10.0).min(1.0);

// Example: Boolean to score
let has_multiple_states = true;
let normalized = if has_multiple_states { 1.0 } else { 0.0 };
```

### Missing Data Handling

If a feature is unavailable:
- Use 0.5 (neutral value) for unknown continuous values
- Use 0.0 (no risk) for unknown boolean values
- Consider the confidence in the prediction may be lower

### Feature Importance

Based on fraud patterns, these features typically have the highest impact:
1. SSN Reuse Score (identity fraud)
2. Employer Verification Status (employment fraud)
3. IP-Address Geographic Match (geographic fraud)
4. Claim Frequency (application fraud)
5. Multi-State Filing Indicator (geographic fraud)

## Model Retraining

To retrain the model with updated fraud patterns:

```rust
use fraudnet::{FraudDataGenerator, NeuralNetwork};

// Generate new training data
let mut fraud_gen = FraudDataGenerator::new(42);
let (train_inputs, train_targets) = fraud_gen.generate_fraud_data(1000, 0.30);
let (test_inputs, test_targets) = fraud_gen.generate_fraud_data(500, 0.30);

// Create and train network
let mut network = NeuralNetwork::new(
    vec![15, 64, 52, 42, 32, 26, 22, 20, 16, 8, 1], 
    0.08, 
    12345
);
network.train(&train_inputs, &train_targets, 1500);

// Evaluate
let accuracy = network.evaluate(&test_inputs, &test_targets, 0.5);
println!("Test Accuracy: {:.2}%", accuracy * 100.0);

// Export
network.save_to_json("model_fraud_detection.json")?;
```

## Performance Metrics

Expected performance on balanced dataset (30% fraud):
- Training Accuracy: >95%
- Testing Accuracy: >90%
- False Positive Rate: <5%
- False Negative Rate: <10%

For production use, monitor:
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1 Score: Harmonic mean of precision and recall

## Integration Checklist

- [ ] Normalize all 15 input features to [0, 1]
- [ ] Handle missing data appropriately
- [ ] Set appropriate risk thresholds for your use case
- [ ] Implement manual review workflow for medium/high risk
- [ ] Log all predictions for audit trail
- [ ] Monitor false positive/negative rates
- [ ] Plan regular model retraining schedule
- [ ] Ensure GDPR/privacy compliance
- [ ] Test with historical fraud cases
- [ ] Set up alerting for high-risk detections

## Support

For detailed information on fraud patterns and features, see [FRAUD_DETECTION.md](FRAUD_DETECTION.md).

For technical details on the neural network architecture, see [README.md](README.md).
