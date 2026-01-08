# Unemployment Insurance Fraud Detection

## Overview

FraudNet is configured to detect Fraud, Waste, and Abuse (FWA) in unemployment insurance systems. This document outlines the fraud detection approach, feature engineering, and data model.

## Common Unemployment Insurance Fraud Schemes

### 1. Identity Fraud
- **Stolen Identity Claims**: Using someone else's personal information to file claims
- **Synthetic Identity**: Creating fake identities with real and fabricated information
- **Indicators**: Multiple claims from same IP/device, mismatched geographic data, rapid succession claims

### 2. Concurrent Employment Fraud
- **Working While Claiming**: Receiving benefits while employed
- **Unreported Earnings**: Failing to report part-time or gig work
- **Indicators**: Claim patterns conflicting with employment records, income spikes

### 3. Benefit Eligibility Fraud
- **False Job Separation**: Lying about reason for job loss
- **Fictitious Employer**: Creating fake employment history
- **Indicators**: Unverifiable employer information, inconsistent work history

### 4. Geographic Fraud
- **Cross-State Claims**: Filing in multiple states simultaneously
- **Residency Fraud**: False address information
- **Indicators**: IP address mismatches, multiple state filings, geographic impossibilities

### 5. Application Fraud
- **Duplicate Claims**: Multiple claims for same period
- **Backdating**: Falsifying claim start dates
- **Indicators**: Temporal anomalies, duplicate SSNs, rapid re-filing

## Neural Network Feature Design

### Input Features (15 features after normalization)

#### Temporal Features (3)
1. **Days Since Last Claim**: Time between claims (normalized 0-1)
2. **Claim Frequency (30 days)**: Number of claims in past 30 days
3. **Time of Day Anomaly**: Unusual filing time (0=normal business hours, 1=suspicious hours)

#### Identity Verification Features (4)
4. **SSN Reuse Score**: How many times SSN appears in recent claims
5. **Age Verification Score**: Age consistency with work history
6. **Address Change Frequency**: Number of address changes in past year
7. **IP Address Reuse Count**: How many claims from same IP in past 30 days

#### Employment History Features (4)
8. **Employer Verification Status**: 0=verified, 1=unverifiable
9. **Employment Duration**: Length of last employment (normalized)
10. **Wage Consistency Score**: How consistent wages are with job type
11. **Separation Reason Risk**: Risk score for termination reason

#### Geographic Features (2)
12. **IP-Address Geographic Match**: Distance between IP location and claimed address
13. **Multi-State Filing Indicator**: 0=single state, 1=multiple states detected

#### Behavioral Features (2)
14. **Document Quality Score**: Quality/authenticity of submitted documents
15. **Response Pattern Anomaly**: Unusual patterns in how questions are answered

### Network Architecture

```
Input Layer: 15 features
    ↓
Hidden Layer 1: 64 neurons (ReLU) - Learn complex feature interactions
    ↓
Hidden Layer 2: 52 neurons (ReLU) - Dimensionality reduction
    ↓
Hidden Layer 3: 42 neurons (ReLU) - Pattern abstraction
    ↓
Hidden Layer 4: 32 neurons (ReLU) - Further refinement
    ↓
Hidden Layer 5: 26 neurons (ReLU) - Compressed representation
    ↓
Hidden Layer 6: 22 neurons (ReLU) - Deep feature extraction
    ↓
Hidden Layer 7: 20 neurons (ReLU) - Subtle pattern detection
    ↓
Hidden Layer 8: 16 neurons (ReLU) - Final refinement
    ↓
Hidden Layer 9: 8 neurons (ReLU) - Distilled features
    ↓
Output Layer: 1 neuron (Sigmoid) - Fraud probability (0-1)
```

### Output Interpretation

- **0.0 - 0.3**: Low risk (legitimate claim)
- **0.3 - 0.6**: Medium risk (flag for review)
- **0.6 - 1.0**: High risk (likely fraud)

## Training Data Patterns

### Legitimate Claims (Class 0)
- Consistent temporal patterns
- Verified employer information
- Geographic consistency
- Normal filing hours
- Single IP per claimant
- Reasonable claim frequency
- Age-appropriate work history

### Fraudulent Claims (Class 1)
- Multiple claims in short time
- Unverifiable employers
- IP/address mismatches
- Off-hours filing patterns
- SSN reuse across claims
- Inconsistent employment history
- Document quality issues
- Multi-state filings

## Data Normalization

All features are normalized to the range [0, 1] or [-1, 1] for optimal neural network performance:

- **Counts**: Divided by reasonable maximum (e.g., 10 claims)
- **Time-based**: Days normalized to [0, 1] scale
- **Boolean flags**: 0 or 1
- **Scores**: Already in [0, 1] range
- **Distances**: Normalized by maximum expected distance

## Implementation Strategy

### Synthetic Data Generation

For training and testing, synthetic fraud patterns are generated:

1. **Identity Fraud Patterns**: High SSN reuse, multiple IPs, address mismatches
2. **Concurrent Employment**: Low employer verification, wage inconsistencies
3. **Geographic Fraud**: IP-address mismatches, multi-state indicators
4. **Application Fraud**: High claim frequency, temporal anomalies
5. **Mixed Fraud**: Combination of multiple fraud types

### Model Training

- **Training Set**: 80% of synthetic data with balanced fraud/legitimate examples
- **Test Set**: 20% held out for validation
- **Epochs**: 1000-2000 for deep network convergence
- **Learning Rate**: 0.05-0.1 with potential decay
- **Batch Processing**: Online learning (one example at a time)

### Performance Metrics

- **Accuracy**: Overall correct classifications
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

## Deployment Considerations

### Client-Side Inference (ONNX Runtime)

The trained model is exported to ONNX format for:
- **Real-time scoring**: Immediate fraud assessment during claim submission
- **Privacy**: No sensitive data sent to servers
- **Scalability**: Client-side processing reduces server load
- **Offline capability**: Can work without constant server connection

### Integration Points

1. **Claim Submission**: Real-time fraud score during application
2. **Batch Processing**: Nightly scoring of all pending claims
3. **Investigator Dashboard**: Review flagged claims with risk scores
4. **Audit Trail**: Log all predictions for compliance

## Regulatory Compliance

### Privacy Protection
- All PII is anonymized during training
- Model doesn't store individual claim data
- GDPR/CCPA compliant processing
- Explainable AI for audit purposes

### Bias Mitigation
- Balanced training data across demographics
- Regular fairness audits
- Human review for high-risk decisions
- Demographic parity monitoring

## Continuous Improvement

### Model Retraining
- Monthly retraining with confirmed fraud cases
- Feedback loop from investigators
- A/B testing of model versions
- Performance monitoring dashboards

### Feature Engineering
- Add new fraud patterns as they emerge
- Remove obsolete features
- Test new data sources
- Validate feature importance

## References

- U.S. Department of Labor - UI Fraud Detection
- National Association of State Workforce Agencies (NASWA)
- Identity Theft Resource Center (ITRC)
- Association of Certified Fraud Examiners (ACFE)
