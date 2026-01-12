/// Database schema for unemployment insurance fraud detection data
/// 
/// This module defines the structure of data expected from the SQL database.
/// The data should be stored in a table with the following columns corresponding
/// to the 15 features used by the neural network.

use serde::{Deserialize, Serialize};

/// Represents a single unemployment insurance claim record from the database
/// Each field corresponds to one of the 15 normalized features (0.0 to 1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimRecord {
    // Metadata (not used as features)
    pub id: Option<i32>,
    pub claim_id: Option<String>,
    
    // Temporal Features (3)
    /// Days since last claim (normalized 0-1, where 1 = 90+ days)
    pub days_since_last_claim: f64,
    
    /// Claim frequency in past 30 days (normalized 0-1, where 1 = 10+ claims)
    pub claim_frequency: f64,
    
    /// Time of day anomaly (0 = normal business hours, 1 = suspicious hours)
    pub time_anomaly: f64,
    
    // Identity Verification Features (4)
    /// SSN reuse score (0-1, where 1 = used in 10+ recent claims)
    pub ssn_reuse_score: f64,
    
    /// Age verification score (0-1, where 1 = major inconsistencies)
    pub age_verification_score: f64,
    
    /// Address change frequency (normalized 0-1, where 1 = 10+ changes/year)
    pub address_change_frequency: f64,
    
    /// IP address reuse count (normalized 0-1, where 1 = 10+ claims from same IP)
    pub ip_reuse_count: f64,
    
    // Employment History Features (4)
    /// Employer verification status (0 = verified, 1 = unverifiable)
    pub employer_verification: f64,
    
    /// Employment duration (normalized 0-1, where 1 = 5+ years)
    pub employment_duration: f64,
    
    /// Wage consistency score (0-1, where 1 = major inconsistencies)
    pub wage_consistency: f64,
    
    /// Separation reason risk (0-1, where 1 = high risk reason)
    pub separation_risk: f64,
    
    // Geographic Features (2)
    /// IP-address geographic match (0-1, where 1 = major mismatch)
    pub ip_address_match: f64,
    
    /// Multi-state filing indicator (0 = single state, 1 = multiple states)
    pub multi_state_filing: f64,
    
    // Behavioral Features (2)
    /// Document quality score (0-1, where 1 = poor/suspicious quality)
    pub document_quality: f64,
    
    /// Response pattern anomaly (0-1, where 1 = highly suspicious pattern)
    pub response_pattern: f64,
    
    // Label
    /// Ground truth label: 0.0 = legitimate, 1.0 = fraudulent
    /// This should be populated based on confirmed fraud investigations
    pub is_fraud: f64,
}

impl ClaimRecord {
    /// Convert a ClaimRecord to a feature vector (Vec<f64>) for the neural network
    /// Returns a vector of 15 features in the expected order
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            // Temporal features (1-3)
            self.days_since_last_claim,
            self.claim_frequency,
            self.time_anomaly,
            
            // Identity features (4-7)
            self.ssn_reuse_score,
            self.age_verification_score,
            self.address_change_frequency,
            self.ip_reuse_count,
            
            // Employment features (8-11)
            self.employer_verification,
            self.employment_duration,
            self.wage_consistency,
            self.separation_risk,
            
            // Geographic features (12-13)
            self.ip_address_match,
            self.multi_state_filing,
            
            // Behavioral features (14-15)
            self.document_quality,
            self.response_pattern,
        ]
    }
    
    /// Get the label (fraud = 1.0, legitimate = 0.0)
    pub fn get_label(&self) -> f64 {
        self.is_fraud
    }
    
    /// Validate that all features are in the valid range [0.0, 1.0]
    pub fn validate(&self) -> Result<(), String> {
        let features = self.to_features();
        for (i, &value) in features.iter().enumerate() {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "Feature {} has invalid value {}: must be in range [0.0, 1.0]",
                    i + 1,
                    value
                ));
            }
        }
        
        if self.is_fraud != 0.0 && self.is_fraud != 1.0 {
            return Err(format!(
                "Label (is_fraud) must be 0.0 or 1.0, got {}",
                self.is_fraud
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod schema_doc {
    //! SQL Schema Documentation
    //! 
    //! Expected database table structure:
    //! 
    //! ```sql
    //! CREATE TABLE unemployment_claims (
    //!     id SERIAL PRIMARY KEY,
    //!     claim_id VARCHAR(50) UNIQUE,
    //!     
    //!     -- Temporal Features
    //!     days_since_last_claim FLOAT NOT NULL CHECK (days_since_last_claim >= 0 AND days_since_last_claim <= 1),
    //!     claim_frequency FLOAT NOT NULL CHECK (claim_frequency >= 0 AND claim_frequency <= 1),
    //!     time_anomaly FLOAT NOT NULL CHECK (time_anomaly >= 0 AND time_anomaly <= 1),
    //!     
    //!     -- Identity Features
    //!     ssn_reuse_score FLOAT NOT NULL CHECK (ssn_reuse_score >= 0 AND ssn_reuse_score <= 1),
    //!     age_verification_score FLOAT NOT NULL CHECK (age_verification_score >= 0 AND age_verification_score <= 1),
    //!     address_change_frequency FLOAT NOT NULL CHECK (address_change_frequency >= 0 AND address_change_frequency <= 1),
    //!     ip_reuse_count FLOAT NOT NULL CHECK (ip_reuse_count >= 0 AND ip_reuse_count <= 1),
    //!     
    //!     -- Employment Features
    //!     employer_verification FLOAT NOT NULL CHECK (employer_verification >= 0 AND employer_verification <= 1),
    //!     employment_duration FLOAT NOT NULL CHECK (employment_duration >= 0 AND employment_duration <= 1),
    //!     wage_consistency FLOAT NOT NULL CHECK (wage_consistency >= 0 AND wage_consistency <= 1),
    //!     separation_risk FLOAT NOT NULL CHECK (separation_risk >= 0 AND separation_risk <= 1),
    //!     
    //!     -- Geographic Features
    //!     ip_address_match FLOAT NOT NULL CHECK (ip_address_match >= 0 AND ip_address_match <= 1),
    //!     multi_state_filing FLOAT NOT NULL CHECK (multi_state_filing >= 0 AND multi_state_filing <= 1),
    //!     
    //!     -- Behavioral Features
    //!     document_quality FLOAT NOT NULL CHECK (document_quality >= 0 AND document_quality <= 1),
    //!     response_pattern FLOAT NOT NULL CHECK (response_pattern >= 0 AND response_pattern <= 1),
    //!     
    //!     -- Label (confirmed fraud status)
    //!     is_fraud FLOAT NOT NULL CHECK (is_fraud IN (0, 1)),
    //!     
    //!     -- Metadata
    //!     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    //!     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    //! );
    //! 
    //! -- Index for efficient querying
    //! CREATE INDEX idx_is_fraud ON unemployment_claims(is_fraud);
    //! CREATE INDEX idx_created_at ON unemployment_claims(created_at);
    //! ```
}
