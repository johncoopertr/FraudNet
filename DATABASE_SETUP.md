# Database Setup Guide for FraudNet

This guide explains how to configure FraudNet to use real data from a PostgreSQL database for training and testing the fraud detection neural network.

## Overview

FraudNet can consume real unemployment insurance claim data from a PostgreSQL database. The data should be stored with 15 normalized features (0.0 to 1.0) that match the neural network's expected input format.

## Prerequisites

- PostgreSQL server (version 9.5 or later)
- Database with unemployment claims data
- Network access to the database server

## Database Schema

Create a table with the following structure:

```sql
CREATE TABLE unemployment_claims (
    id SERIAL PRIMARY KEY,
    claim_id VARCHAR(50) UNIQUE,
    
    -- Temporal Features (3 features)
    days_since_last_claim FLOAT NOT NULL CHECK (days_since_last_claim >= 0 AND days_since_last_claim <= 1),
    claim_frequency FLOAT NOT NULL CHECK (claim_frequency >= 0 AND claim_frequency <= 1),
    time_anomaly FLOAT NOT NULL CHECK (time_anomaly >= 0 AND time_anomaly <= 1),
    
    -- Identity Features (4 features)
    ssn_reuse_score FLOAT NOT NULL CHECK (ssn_reuse_score >= 0 AND ssn_reuse_score <= 1),
    age_verification_score FLOAT NOT NULL CHECK (age_verification_score >= 0 AND age_verification_score <= 1),
    address_change_frequency FLOAT NOT NULL CHECK (address_change_frequency >= 0 AND address_change_frequency <= 1),
    ip_reuse_count FLOAT NOT NULL CHECK (ip_reuse_count >= 0 AND ip_reuse_count <= 1),
    
    -- Employment Features (4 features)
    employer_verification FLOAT NOT NULL CHECK (employer_verification >= 0 AND employer_verification <= 1),
    employment_duration FLOAT NOT NULL CHECK (employment_duration >= 0 AND employment_duration <= 1),
    wage_consistency FLOAT NOT NULL CHECK (wage_consistency >= 0 AND wage_consistency <= 1),
    separation_risk FLOAT NOT NULL CHECK (separation_risk >= 0 AND separation_risk <= 1),
    
    -- Geographic Features (2 features)
    ip_address_match FLOAT NOT NULL CHECK (ip_address_match >= 0 AND ip_address_match <= 1),
    multi_state_filing FLOAT NOT NULL CHECK (multi_state_filing >= 0 AND multi_state_filing <= 1),
    
    -- Behavioral Features (2 features)
    document_quality FLOAT NOT NULL CHECK (document_quality >= 0 AND document_quality <= 1),
    response_pattern FLOAT NOT NULL CHECK (response_pattern >= 0 AND response_pattern <= 1),
    
    -- Label (confirmed fraud status)
    is_fraud FLOAT NOT NULL CHECK (is_fraud IN (0, 1)),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX idx_is_fraud ON unemployment_claims(is_fraud);
CREATE INDEX idx_created_at ON unemployment_claims(created_at);
```

## Feature Descriptions

All features should be normalized to the range [0.0, 1.0]:

### Temporal Features (3)
1. **days_since_last_claim**: Time between claims (0 = 0 days, 1 = 90+ days)
2. **claim_frequency**: Number of claims in past 30 days (0 = 0 claims, 1 = 10+ claims)
3. **time_anomaly**: Filing time (0 = normal business hours, 1 = suspicious hours)

### Identity Verification Features (4)
4. **ssn_reuse_score**: SSN usage in recent claims (0 = unique, 1 = 10+ uses)
5. **age_verification_score**: Age consistency with work history (0 = verified, 1 = major issues)
6. **address_change_frequency**: Address changes in past year (0 = stable, 1 = 10+ changes)
7. **ip_reuse_count**: Claims from same IP in past 30 days (0 = unique, 1 = 10+ claims)

### Employment History Features (4)
8. **employer_verification**: Employer verification status (0 = verified, 1 = unverifiable)
9. **employment_duration**: Length of last employment (0 = 0 months, 1 = 5+ years)
10. **wage_consistency**: Wage consistency with job type (0 = consistent, 1 = major issues)
11. **separation_risk**: Risk score for termination reason (0 = low risk, 1 = high risk)

### Geographic Features (2)
12. **ip_address_match**: Distance between IP and claimed address (0 = match, 1 = major mismatch)
13. **multi_state_filing**: Multi-state filing indicator (0 = single state, 1 = multiple states)

### Behavioral Features (2)
14. **document_quality**: Quality/authenticity of documents (0 = good, 1 = poor/suspicious)
15. **response_pattern**: Pattern in how questions are answered (0 = normal, 1 = suspicious)

### Label
- **is_fraud**: Ground truth (0.0 = legitimate, 1.0 = fraudulent)

## Configuration

### 1. Copy the Example Environment File

```bash
cp .env.example .env
```

### 2. Edit .env File

Open `.env` and configure your database connection:

```env
# PostgreSQL Connection String
DATABASE_URL=postgres://username:password@hostname:5432/database_name

# Data Split Configuration (must sum to 1.0)
TRAIN_SPLIT=0.7   # 70% for training
TEST_SPLIT=0.2    # 20% for testing
DEMO_SPLIT=0.1    # 10% for demo

# Number of demo records to export for web demonstration
DEMO_RECORD_COUNT=10
```

**Security Note:** The `.env` file is automatically ignored by git to keep your credentials secure. Never commit this file to version control.

### 3. Build with Database Feature

FraudNet uses optional database features. Build with database support:

```bash
cargo build --features database
```

Or run directly:

```bash
cargo run --features database
```

## Usage

### Training with Real Data

Once configured, simply run FraudNet:

```bash
cargo run --features database
```

FraudNet will:
1. Load environment variables from `.env`
2. Connect to your PostgreSQL database
3. Fetch all records from the `unemployment_claims` table
4. Validate all data (ensuring values are in [0.0, 1.0] range)
5. Split data into training (70%), testing (20%), and demo (10%) sets
6. Train the neural network on the real data
7. Export demo records to `demo_data.json` for web demonstration
8. Export the trained model to `model_fraud_detection.json`

### Fallback to Synthetic Data

If the database is not configured or connection fails, FraudNet will automatically fall back to synthetic data generation with a clear message:

```
No database configuration found (.env file not present)
Using synthetic data for training and testing...
```

This ensures the system always works, even without database access.

## Data Quality Guidelines

For best training results:

1. **Balanced Dataset**: Include roughly equal numbers of fraudulent and legitimate claims
2. **Sufficient Size**: Aim for at least 1,000 records (more is better)
3. **Variety**: Include diverse fraud patterns (identity, employment, geographic, etc.)
4. **Accuracy**: Ensure `is_fraud` labels are from confirmed investigations
5. **Normalized Values**: All features must be properly normalized to [0.0, 1.0]

## Troubleshooting

### Connection Errors

**Error:** `Failed to connect to database`

**Solutions:**
- Verify DATABASE_URL format: `postgres://user:password@host:port/database`
- Check network connectivity to database server
- Ensure database server is running
- Verify firewall rules allow connections
- Check username and password are correct

### No Data Found

**Error:** `No valid records found in database`

**Solutions:**
- Verify the table `unemployment_claims` exists
- Check table has data: `SELECT COUNT(*) FROM unemployment_claims;`
- Ensure data meets validation requirements (all features in [0.0, 1.0])

### Invalid Data Values

**Error:** `Feature X has invalid value`

**Solutions:**
- Review feature normalization
- Check for NULL values in required fields
- Verify CHECK constraints are properly applied
- Use SQL queries to find problematic records:
  ```sql
  SELECT * FROM unemployment_claims 
  WHERE days_since_last_claim < 0 OR days_since_last_claim > 1;
  ```

## Example Data Insertion

Here's an example of inserting a legitimate claim:

```sql
INSERT INTO unemployment_claims (
    claim_id,
    days_since_last_claim, claim_frequency, time_anomaly,
    ssn_reuse_score, age_verification_score, address_change_frequency, ip_reuse_count,
    employer_verification, employment_duration, wage_consistency, separation_risk,
    ip_address_match, multi_state_filing,
    document_quality, response_pattern,
    is_fraud
) VALUES (
    'CLAIM_001',
    0.5, 0.1, 0.2,           -- Temporal: normal patterns
    0.0, 0.1, 0.1, 0.05,     -- Identity: verified
    0.0, 0.7, 0.1, 0.2,      -- Employment: good history
    0.1, 0.0,                 -- Geographic: consistent
    0.2, 0.1,                 -- Behavioral: normal
    0.0                       -- Legitimate claim
);
```

And an example of a fraudulent claim:

```sql
INSERT INTO unemployment_claims (
    claim_id,
    days_since_last_claim, claim_frequency, time_anomaly,
    ssn_reuse_score, age_verification_score, address_change_frequency, ip_reuse_count,
    employer_verification, employment_duration, wage_consistency, separation_risk,
    ip_address_match, multi_state_filing,
    document_quality, response_pattern,
    is_fraud
) VALUES (
    'CLAIM_002',
    0.1, 0.8, 0.9,           -- Temporal: suspicious patterns
    0.9, 0.8, 0.7, 0.9,      -- Identity: red flags
    0.6, 0.3, 0.6, 0.7,      -- Employment: inconsistencies
    0.7, 0.8,                 -- Geographic: mismatches
    0.8, 0.9,                 -- Behavioral: very suspicious
    1.0                       -- Confirmed fraud
);
```

## Security Best Practices

1. **Database User Permissions**: Create a read-only database user for FraudNet
   ```sql
   CREATE USER fraudnet_reader WITH PASSWORD 'secure_password';
   GRANT SELECT ON unemployment_claims TO fraudnet_reader;
   ```

2. **SSL Connections**: Use SSL for database connections in production:
   ```env
   DATABASE_URL=postgres://user:pass@host:5432/db?sslmode=require
   ```

3. **Firewall**: Restrict database access to specific IP addresses

4. **Password Management**: Use strong passwords and rotate them regularly

5. **Environment Variables**: Never commit `.env` file to version control

## Next Steps

After setting up the database:

1. Train the model: `cargo run --features database`
2. Convert to ONNX: `python3 scripts/json_to_onnx.py`
3. Test in browser: `cargo test-web`
4. Review demo data: Check `demo_data.json` for exported records
5. Deploy the model for real-time fraud detection

See [FRAUD_DETECTION.md](FRAUD_DETECTION.md) for more details on the fraud detection approach and features.
