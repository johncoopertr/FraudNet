-- Sample Data Insertion Script for FraudNet
-- This script provides example data to test the database integration
-- 
-- Usage:
--   psql -U username -d database_name -f sample_data.sql

-- First, ensure the table exists (from DATABASE_SETUP.md)
-- If it doesn't exist, run the CREATE TABLE statement first

-- Insert some legitimate claims
INSERT INTO unemployment_claims (
    claim_id,
    days_since_last_claim, claim_frequency, time_anomaly,
    ssn_reuse_score, age_verification_score, address_change_frequency, ip_reuse_count,
    employer_verification, employment_duration, wage_consistency, separation_risk,
    ip_address_match, multi_state_filing,
    document_quality, response_pattern,
    is_fraud
) VALUES
    -- Legitimate Claim 1: Clean record, verified employer
    ('LEGIT_001', 
     0.5, 0.1, 0.2,           -- Temporal: normal patterns
     0.0, 0.1, 0.1, 0.05,     -- Identity: verified
     0.0, 0.7, 0.1, 0.2,      -- Employment: good history
     0.1, 0.0,                 -- Geographic: consistent
     0.2, 0.1,                 -- Behavioral: normal
     0.0),                     -- Label: Legitimate

    -- Legitimate Claim 2: Another clean record
    ('LEGIT_002',
     0.6, 0.0, 0.3,
     0.05, 0.15, 0.05, 0.0,
     0.05, 0.8, 0.05, 0.15,
     0.05, 0.0,
     0.15, 0.05,
     0.0),

    -- Legitimate Claim 3: Slightly older claim
    ('LEGIT_003',
     0.7, 0.15, 0.25,
     0.0, 0.2, 0.15, 0.1,
     0.1, 0.6, 0.2, 0.25,
     0.15, 0.0,
     0.25, 0.15,
     0.0),

    -- Fraudulent Claim 1: Identity fraud - high SSN reuse, multiple IPs
    ('FRAUD_001',
     0.1, 0.8, 0.9,           -- Temporal: rapid filing, off hours
     0.9, 0.8, 0.7, 0.9,      -- Identity: major red flags
     0.6, 0.3, 0.6, 0.7,      -- Employment: inconsistencies
     0.7, 0.8,                 -- Geographic: mismatches
     0.8, 0.9,                 -- Behavioral: very suspicious
     1.0),                     -- Label: Fraudulent

    -- Fraudulent Claim 2: Employment fraud - working while claiming
    ('FRAUD_002',
     0.3, 0.4, 0.2,           -- Temporal: regular (hiding)
     0.1, 0.1, 0.1, 0.1,      -- Identity: legitimate person
     0.9, 0.1, 0.9, 0.8,      -- Employment: hiding work
     0.2, 0.1,                 -- Geographic: consistent
     0.3, 0.7,                 -- Behavioral: evasive
     1.0),

    -- Fraudulent Claim 3: Geographic fraud - multi-state filing
    ('FRAUD_003',
     0.2, 0.7, 0.8,
     0.5, 0.4, 0.6, 0.7,
     0.5, 0.4, 0.5, 0.6,
     0.9, 0.9,                 -- Geographic: HIGH RISK
     0.7, 0.8,
     1.0),

    -- Legitimate Claim 4: Longer tenure, good verification
    ('LEGIT_004',
     0.65, 0.05, 0.15,
     0.0, 0.05, 0.0, 0.0,
     0.0, 0.85, 0.05, 0.1,
     0.05, 0.0,
     0.1, 0.05,
     0.0),

    -- Fraudulent Claim 4: Application fraud - duplicate claims, backdating
    ('FRAUD_004',
     0.05, 0.95, 0.95,        -- Temporal: very suspicious
     0.3, 0.2, 0.3, 0.6,
     0.6, 0.5, 0.6, 0.7,
     0.4, 0.3,
     0.8, 0.9,                 -- Behavioral: poor docs, bad patterns
     1.0),

    -- Legitimate Claim 5: Medium tenure, normal patterns
    ('LEGIT_005',
     0.55, 0.12, 0.18,
     0.05, 0.12, 0.08, 0.05,
     0.08, 0.65, 0.15, 0.18,
     0.08, 0.0,
     0.18, 0.12,
     0.0),

    -- Fraudulent Claim 5: Mixed fraud - multiple red flags
    ('FRAUD_005',
     0.15, 0.85, 0.88,
     0.85, 0.75, 0.8, 0.85,
     0.75, 0.4, 0.7, 0.8,
     0.75, 0.7,
     0.85, 0.9,
     1.0),

    -- Add more legitimate claims for balance
    ('LEGIT_006', 0.48, 0.08, 0.22, 0.02, 0.08, 0.05, 0.03, 0.05, 0.72, 0.08, 0.15, 0.08, 0.0, 0.15, 0.08, 0.0),
    ('LEGIT_007', 0.62, 0.06, 0.19, 0.0, 0.11, 0.12, 0.08, 0.06, 0.68, 0.12, 0.19, 0.11, 0.0, 0.19, 0.11, 0.0),
    ('LEGIT_008', 0.58, 0.11, 0.21, 0.04, 0.14, 0.09, 0.06, 0.09, 0.64, 0.16, 0.21, 0.14, 0.0, 0.21, 0.14, 0.0),
    ('LEGIT_009', 0.52, 0.09, 0.17, 0.03, 0.09, 0.07, 0.04, 0.07, 0.69, 0.13, 0.17, 0.12, 0.0, 0.17, 0.09, 0.0),
    ('LEGIT_010', 0.67, 0.04, 0.16, 0.01, 0.06, 0.04, 0.02, 0.04, 0.81, 0.09, 0.14, 0.09, 0.0, 0.14, 0.06, 0.0);

-- Verify the data was inserted
SELECT 
    COUNT(*) as total_claims,
    SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END) as legitimate_claims,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraudulent_claims
FROM unemployment_claims;

-- Show a sample of the data
SELECT 
    claim_id,
    days_since_last_claim,
    ssn_reuse_score,
    employer_verification,
    is_fraud
FROM unemployment_claims
ORDER BY is_fraud DESC, claim_id
LIMIT 10;
