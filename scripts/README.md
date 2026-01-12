# Database Scripts for FraudNet

This directory contains SQL scripts to help set up and populate the Microsoft SQL Server database for FraudNet fraud detection training.

## Files

- **`create_schema.sql`** - Creates the `unemployment_claims` table with all required features and indexes
- **`sample_data.sql`** - Inserts sample test data (10 legitimate claims + 5 fraudulent claims)
- **`json_to_onnx.py`** - Converts trained neural network models from JSON to ONNX format

## Quick Start

### 1. Create the Database

First, create a Microsoft SQL Server database for FraudNet:

```bash
# Create database (if it doesn't exist)
sqlcmd -Q "CREATE DATABASE fraudnet

# Or using psql
sqlcmd -U postgres -c "CREATE DATABASE fraudnet;"
```

### 2. Create the Schema

Run the schema creation script:

```bash
sqlcmd -U username -d fraudnet -f scripts/create_schema.sql
```

This will:
- Create the `unemployment_claims` table
- Add CHECK constraints to ensure all features are in [0.0, 1.0] range
- Create indexes for efficient querying
- Set up automatic `updated_at` timestamp updates

### 3. Insert Sample Data

Load some test data to verify the setup:

```bash
sqlcmd -U username -d fraudnet -f scripts/sample_data.sql
```

This inserts:
- 10 legitimate unemployment claims
- 5 fraudulent claims (various fraud types)

### 4. Verify the Data

Check that the data was loaded correctly:

```sql
sqlcmd -U username -d fraudnet

-- Count records
SELECT COUNT(*) FROM unemployment_claims;

-- View fraud distribution
SELECT 
    is_fraud,
    COUNT(*) as count
FROM unemployment_claims
GROUP BY is_fraud;

-- View sample records
SELECT * FROM unemployment_claims LIMIT 5;
```

## Feature Descriptions

The `unemployment_claims` table has 15 features (all normalized to 0.0-1.0):

### Temporal Features (3)
1. **days_since_last_claim** - Time between claims (0=0 days, 1=90+ days)
2. **claim_frequency** - Claims in past 30 days (0=0 claims, 1=10+ claims)
3. **time_anomaly** - Filing time (0=business hours, 1=suspicious hours)

### Identity Features (4)
4. **ssn_reuse_score** - SSN usage in recent claims (0=unique, 1=10+ uses)
5. **age_verification_score** - Age consistency (0=verified, 1=issues)
6. **address_change_frequency** - Address changes/year (0=stable, 1=10+ changes)
7. **ip_reuse_count** - Claims from same IP (0=unique, 1=10+ claims)

### Employment Features (4)
8. **employer_verification** - Employer status (0=verified, 1=unverifiable)
9. **employment_duration** - Last employment length (0=0 months, 1=5+ years)
10. **wage_consistency** - Wage consistency (0=consistent, 1=issues)
11. **separation_risk** - Termination reason risk (0=low, 1=high)

### Geographic Features (2)
12. **ip_address_match** - IP/address distance (0=match, 1=mismatch)
13. **multi_state_filing** - Multi-state indicator (0=single, 1=multiple)

### Behavioral Features (2)
14. **document_quality** - Document quality (0=good, 1=suspicious)
15. **response_pattern** - Answer patterns (0=normal, 1=suspicious)

### Label
- **is_fraud** - Ground truth (0.0=legitimate, 1.0=fraudulent)

## Inserting Your Own Data

To insert your own data, use this template:

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
    'YOUR_CLAIM_ID',
    0.5, 0.1, 0.2,  -- Temporal features
    0.0, 0.1, 0.1, 0.05,  -- Identity features
    0.0, 0.7, 0.1, 0.2,  -- Employment features
    0.1, 0.0,  -- Geographic features
    0.2, 0.1,  -- Behavioral features
    0.0  -- Label (0=legit, 1=fraud)
);
```

**Important:** All feature values must be between 0.0 and 1.0, and `is_fraud` must be either 0.0 or 1.0.

## Security Recommendations

### Create a Read-Only User

For production use, create a dedicated read-only database user for FraudNet:

```sql
-- Create read-only user
CREATE USER fraudnet_reader WITH PASSWORD 'secure_password_here';

-- Grant SELECT permission
GRANT CONNECT ON DATABASE fraudnet TO fraudnet_reader;
GRANT SELECT ON unemployment_claims TO fraudnet_reader;
```

Then use this user in your `.env` file:

```env
DATABASE_URL=postgres://fraudnet_reader:secure_password_here@localhost:5432/fraudnet
```

### Enable SSL Connections

For remote databases, always use SSL:

```env
DATABASE_URL=postgres://user:pass@host:5432/fraudnet?sslmode=require
```

## Training with the Data

Once your database is set up:

1. Configure `.env` with your database credentials
2. Run FraudNet with database support:
   ```bash
   cargo run --features database
   ```
3. The program will:
   - Load data from Microsoft SQL Server
   - Split it into training/testing/demo sets
   - Train the neural network
   - Export the trained model and demo data

See [DATABASE_SETUP.md](../DATABASE_SETUP.md) for complete setup instructions.

## Troubleshooting

### Permission Denied

If you get permission errors:
```sql
GRANT ALL PRIVILEGES ON DATABASE fraudnet TO username;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO username;
```

### Connection Failed

Verify Microsoft SQL Server is running:
```bash
# Check if Microsoft SQL Server is running
pg_isready

# Check Microsoft SQL Server status
systemctl status postgresql  # Linux
brew services list  # macOS
```

### Invalid Data

If inserts fail due to CHECK constraints, verify all values are in [0.0, 1.0] range and `is_fraud` is either 0.0 or 1.0.

## Additional Resources

- [Microsoft SQL Server Documentation](https://www.postgresql.org/docs/)
- [FraudNet Database Setup Guide](../DATABASE_SETUP.md)
- [FraudNet Fraud Detection Documentation](../FRAUD_DETECTION.md)
