/// Database loader for fetching unemployment insurance claim data
/// 
/// This module handles connecting to a PostgreSQL database and loading
/// claim records for training, testing, and demo purposes.

#[cfg(feature = "database")]
use postgres::{Client, NoTls};

use crate::db_schema::ClaimRecord;
use crate::matrix::Matrix;
use std::env;

/// Configuration for database connection
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub connection_string: String,
    pub train_split: f64,
    pub test_split: f64,
    pub demo_split: f64,
    pub demo_record_count: usize,
    pub demo_output_file: String,
}

impl DatabaseConfig {
    /// Load configuration from environment variables
    /// 
    /// Expected environment variables:
    /// - DATABASE_URL: PostgreSQL connection string
    /// - TRAIN_SPLIT: Percentage of data for training (default: 0.7)
    /// - TEST_SPLIT: Percentage of data for testing (default: 0.2)
    /// - DEMO_SPLIT: Percentage of data for demo (default: 0.1)
    /// - DEMO_RECORD_COUNT: Number of demo records to export (default: 10)
    /// - DEMO_OUTPUT_FILE: Filename for demo data export (default: demo_data.json)
    pub fn from_env() -> Result<Self, String> {
        let connection_string = env::var("DATABASE_URL")
            .map_err(|_| "DATABASE_URL not found in environment. Please set it in .env file.".to_string())?;
        
        let train_split = env::var("TRAIN_SPLIT")
            .unwrap_or_else(|_| "0.7".to_string())
            .parse::<f64>()
            .map_err(|_| "Invalid TRAIN_SPLIT value".to_string())?;
        
        let test_split = env::var("TEST_SPLIT")
            .unwrap_or_else(|_| "0.2".to_string())
            .parse::<f64>()
            .map_err(|_| "Invalid TEST_SPLIT value".to_string())?;
        
        let demo_split = env::var("DEMO_SPLIT")
            .unwrap_or_else(|_| "0.1".to_string())
            .parse::<f64>()
            .map_err(|_| "Invalid DEMO_SPLIT value".to_string())?;
        
        let demo_record_count = env::var("DEMO_RECORD_COUNT")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<usize>()
            .map_err(|_| "Invalid DEMO_RECORD_COUNT value".to_string())?;
        
        let demo_output_file = env::var("DEMO_OUTPUT_FILE")
            .unwrap_or_else(|_| "demo_data.json".to_string());
        
        // Validate splits sum to approximately 1.0
        // Tolerance of 0.01 allows for minor floating-point rounding differences
        // while ensuring splits are reasonably close to 100%
        const SPLIT_TOLERANCE: f64 = 0.01;
        let total = train_split + test_split + demo_split;
        if (total - 1.0).abs() > SPLIT_TOLERANCE {
            return Err(format!(
                "Train, test, and demo splits must sum to 1.0, got {}",
                total
            ));
        }
        
        Ok(DatabaseConfig {
            connection_string,
            train_split,
            test_split,
            demo_split,
            demo_record_count,
            demo_output_file,
        })
    }
}

/// Split dataset into training, testing, and demo sets
pub fn split_data(
    records: Vec<ClaimRecord>,
    config: &DatabaseConfig,
) -> (Vec<ClaimRecord>, Vec<ClaimRecord>, Vec<ClaimRecord>) {
    let total = records.len();
    let train_end = (total as f64 * config.train_split) as usize;
    let test_end = train_end + (total as f64 * config.test_split) as usize;
    
    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut demo = Vec::new();
    
    for (i, record) in records.into_iter().enumerate() {
        if i < train_end {
            train.push(record);
        } else if i < test_end {
            test.push(record);
        } else {
            demo.push(record);
        }
    }
    
    // Limit demo records to configured count
    demo.truncate(config.demo_record_count);
    
    (train, test, demo)
}

/// Convert claim records to neural network format
pub fn records_to_training_data(records: &[ClaimRecord]) -> (Vec<Matrix>, Vec<Matrix>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for record in records {
        let features: Vec<f64> = record.to_features();
        let input = Matrix::from_vec(15, 1, features);
        let target = Matrix::from_vec(1, 1, vec![record.get_label()]);
        
        inputs.push(input);
        targets.push(target);
    }
    
    (inputs, targets)
}

#[cfg(feature = "database")]
/// Load claim records from PostgreSQL database
/// 
/// Note: This function uses NoTls for simplicity. For production use with remote databases,
/// consider using SSL/TLS by:
/// 1. Adding `postgres-native-tls` or `postgres-openssl` as a dependency
/// 2. Replacing NoTls with a TLS connector
/// 3. Using a connection string like: postgres://user:pass@host:5432/db?sslmode=require
pub fn load_from_database(config: &DatabaseConfig) -> Result<Vec<ClaimRecord>, String> {
    // Connect to the database (NoTls - for production, consider using TLS)
    let mut client = Client::connect(&config.connection_string, NoTls)
        .map_err(|e| format!("Failed to connect to database: {}", e))?;
    
    println!("Connected to database successfully");
    
    // Query all records
    let query = "
        SELECT 
            id, claim_id,
            days_since_last_claim, claim_frequency, time_anomaly,
            ssn_reuse_score, age_verification_score, address_change_frequency, ip_reuse_count,
            employer_verification, employment_duration, wage_consistency, separation_risk,
            ip_address_match, multi_state_filing,
            document_quality, response_pattern,
            is_fraud
        FROM unemployment_claims
        ORDER BY id
    ";
    
    let rows = client.query(query, &[])
        .map_err(|e| format!("Failed to query database: {}", e))?;
    
    println!("Fetched {} records from database", rows.len());
    
    let mut records = Vec::new();
    let mut skipped_count = 0;
    
    for row in rows {
        let record = ClaimRecord {
            id: row.get(0),
            claim_id: row.get(1),
            days_since_last_claim: row.get(2),
            claim_frequency: row.get(3),
            time_anomaly: row.get(4),
            ssn_reuse_score: row.get(5),
            age_verification_score: row.get(6),
            address_change_frequency: row.get(7),
            ip_reuse_count: row.get(8),
            employer_verification: row.get(9),
            employment_duration: row.get(10),
            wage_consistency: row.get(11),
            separation_risk: row.get(12),
            ip_address_match: row.get(13),
            multi_state_filing: row.get(14),
            document_quality: row.get(15),
            response_pattern: row.get(16),
            is_fraud: row.get(17),
        };
        
        // Validate record
        if let Err(e) = record.validate() {
            eprintln!("Warning: Skipping invalid record {}: {}", 
                record.claim_id.as_deref().unwrap_or("unknown"), 
                e
            );
            skipped_count += 1;
            continue;
        }
        
        records.push(record);
    }
    
    if skipped_count > 0 {
        println!("Skipped {} invalid records", skipped_count);
    }
    
    if records.is_empty() {
        return Err("No valid records found in database".to_string());
    }
    
    println!("Loaded {} valid records", records.len());
    
    Ok(records)
}

#[cfg(not(feature = "database"))]
/// Stub function when database feature is not enabled
pub fn load_from_database(_config: &DatabaseConfig) -> Result<Vec<ClaimRecord>, String> {
    Err("Database feature not enabled. Rebuild with --features database".to_string())
}

/// Export demo data to JSON file for web demonstration
pub fn export_demo_data(records: &[ClaimRecord], filename: &str) -> Result<(), String> {
    let json = serde_json::to_string_pretty(records)
        .map_err(|e| format!("Failed to serialize demo data: {}", e))?;
    
    std::fs::write(filename, json)
        .map_err(|e| format!("Failed to write demo data to {}: {}", filename, e))?;
    
    println!("✓ Exported {} demo records to {}", records.len(), filename);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_split_data() {
        let records: Vec<ClaimRecord> = (0..100).map(|i| ClaimRecord {
            id: Some(i),
            claim_id: Some(format!("CLAIM_{}", i)),
            days_since_last_claim: 0.5,
            claim_frequency: 0.1,
            time_anomaly: 0.2,
            ssn_reuse_score: 0.0,
            age_verification_score: 0.1,
            address_change_frequency: 0.1,
            ip_reuse_count: 0.05,
            employer_verification: 0.0,
            employment_duration: 0.7,
            wage_consistency: 0.1,
            separation_risk: 0.2,
            ip_address_match: 0.1,
            multi_state_filing: 0.0,
            document_quality: 0.2,
            response_pattern: 0.1,
            is_fraud: if i % 3 == 0 { 1.0 } else { 0.0 },
        }).collect();
        
        let config = DatabaseConfig {
            connection_string: "unused".to_string(),
            train_split: 0.7,
            test_split: 0.2,
            demo_split: 0.1,
            demo_record_count: 5,
            demo_output_file: "demo_data.json".to_string(),
        };
        
        let (train, test, demo) = split_data(records, &config);
        
        assert_eq!(train.len(), 70);
        assert_eq!(test.len(), 20);
        assert_eq!(demo.len(), 5); // Limited by demo_record_count
    }
    
    #[test]
    fn test_records_to_training_data() {
        let records = vec![
            ClaimRecord {
                id: Some(1),
                claim_id: Some("CLAIM_1".to_string()),
                days_since_last_claim: 0.5,
                claim_frequency: 0.1,
                time_anomaly: 0.2,
                ssn_reuse_score: 0.0,
                age_verification_score: 0.1,
                address_change_frequency: 0.1,
                ip_reuse_count: 0.05,
                employer_verification: 0.0,
                employment_duration: 0.7,
                wage_consistency: 0.1,
                separation_risk: 0.2,
                ip_address_match: 0.1,
                multi_state_filing: 0.0,
                document_quality: 0.2,
                response_pattern: 0.1,
                is_fraud: 0.0,
            },
        ];
        
        let (inputs, targets) = records_to_training_data(&records);
        
        assert_eq!(inputs.len(), 1);
        assert_eq!(targets.len(), 1);
        assert_eq!(inputs[0].rows, 15);
        assert_eq!(inputs[0].cols, 1);
        assert_eq!(targets[0].get(0, 0), 0.0);
    }
}
