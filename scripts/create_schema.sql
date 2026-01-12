-- Database Schema Creation Script for FraudNet (Microsoft SQL Server)
-- Creates the unemployment_claims table with all required features
--
-- Usage:
--   sqlcmd -S server_name -d database_name -i create_schema.sql
--   Or execute in SQL Server Management Studio

-- Drop table if it exists (be careful in production!)
-- WARNING: This will DELETE ALL DATA in the unemployment_claims table!
-- Only uncomment these lines if you are absolutely sure you want to recreate the table
-- IF OBJECT_ID('unemployment_claims', 'U') IS NOT NULL
--     DROP TABLE unemployment_claims;
-- GO

-- Create the unemployment claims table
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'unemployment_claims') AND type in (N'U'))
BEGIN
    CREATE TABLE unemployment_claims (
        -- Primary key and identifier
        id INT IDENTITY(1,1) PRIMARY KEY,
        claim_id VARCHAR(50) UNIQUE NOT NULL,
        
        -- Temporal Features (3 features)
        -- All values normalized to [0.0, 1.0]
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
        
        -- Metadata (optional but recommended)
        created_at DATETIME2 DEFAULT GETDATE(),
        updated_at DATETIME2 DEFAULT GETDATE()
    );
END
GO

-- Create indexes for efficient querying
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_is_fraud' AND object_id = OBJECT_ID('unemployment_claims'))
    CREATE INDEX idx_is_fraud ON unemployment_claims(is_fraud);
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_created_at' AND object_id = OBJECT_ID('unemployment_claims'))
    CREATE INDEX idx_created_at ON unemployment_claims(created_at);
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_claim_id' AND object_id = OBJECT_ID('unemployment_claims'))
    CREATE INDEX idx_claim_id ON unemployment_claims(claim_id);
GO

-- Create a trigger to update the updated_at timestamp
IF OBJECT_ID('tr_update_unemployment_claims_updated_at', 'TR') IS NOT NULL
    DROP TRIGGER tr_update_unemployment_claims_updated_at;
GO

CREATE TRIGGER tr_update_unemployment_claims_updated_at
ON unemployment_claims
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE unemployment_claims
    SET updated_at = GETDATE()
    FROM unemployment_claims u
    INNER JOIN inserted i ON u.id = i.id;
END
GO

-- Display table structure
EXEC sp_help 'unemployment_claims';
GO

-- Show that the table is empty
SELECT COUNT(*) as total_records FROM unemployment_claims;
GO

-- Display feature documentation
SELECT 
    'Schema created successfully!' as status,
    '15 features + 1 label + metadata' as structure,
    'Run scripts/sample_data.sql to insert test data' as next_step;
GO
