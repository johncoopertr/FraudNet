/// Database loader for fetching unemployment insurance claim data
/// 
/// This module handles connecting to a Microsoft SQL Server database and loading
/// claim records for training, testing, and demo purposes.

#[cfg(feature = "database")]
use tiberius::{Client, Config};
#[cfg(feature = "database")]
use tokio::net::TcpStream;
#[cfg(feature = "database")]
use tokio_util::compat::TokioAsyncWriteCompatExt;

use crate::db_schema::ClaimRecord;
use crate::matrix::Matrix;
use std::env;

/// Tolerance for validating that data splits sum to 1.0
/// Allows for minor floating-point rounding differences
const SPLIT_TOLERANCE: f64 = 0.01;

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
    /// - DATABASE_URL: SQL Server connection string
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
        // Using module-level constant SPLIT_TOLERANCE
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
/// 
/// Note: This function performs a sequential split (not randomized).
/// Records are split in the order they are provided. If your data
/// is ordered by time or other factors, consider shuffling before
/// calling this function to avoid temporal or ordering bias.
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
/// Load claim records from Microsoft SQL Server database
/// 
/// Note: This function uses TLS by default with rustls for secure connections.
/// Connection string format: Server=hostname;Database=dbname;User Id=username;Password=password;TrustServerCertificate=true
pub fn load_from_database(config: &DatabaseConfig) -> Result<Vec<ClaimRecord>, String> {
    // Parse connection string
    let mut cfg = Config::from_ado_string(&config.connection_string)
        .map_err(|e| format!("Failed to parse connection string: {}", e))?;
    
    // Enable encryption
    cfg.encryption(tiberius::EncryptionLevel::Required);
    cfg.trust_cert();
    
    // Create tokio runtime for async operations
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| format!("Failed to create runtime: {}", e))?;
    
    rt.block_on(async {
        // Connect to SQL Server
        let tcp = TcpStream::connect(cfg.get_addr())
            .await
            .map_err(|e| format!("Failed to connect to SQL Server: {}", e))?;
        
        tcp.set_nodelay(true)
            .map_err(|e| format!("Failed to set nodelay: {}", e))?;
        
        let mut client = Client::connect(cfg, tcp.compat_write())
            .await
            .map_err(|e| format!("Failed to authenticate with SQL Server: {}", e))?;
        
        println!("Connected to SQL Server successfully");
        
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
        
        let stream = client.query(query, &[])
            .await
            .map_err(|e| format!("Failed to query database: {}", e))?;
        
        let rows = stream.into_first_result()
            .await
            .map_err(|e| format!("Failed to fetch results: {}", e))?;
        
        println!("Fetched {} records from database", rows.len());
        
        let mut records = Vec::new();
        let mut skipped_count = 0;
        
        for row in rows {
            let record = ClaimRecord {
                id: row.get::<i32, _>(0),
                claim_id: row.get::<&str, _>(1).map(|s| s.to_string()),
                days_since_last_claim: row.get::<f64, _>(2).unwrap_or(0.0),
                claim_frequency: row.get::<f64, _>(3).unwrap_or(0.0),
                time_anomaly: row.get::<f64, _>(4).unwrap_or(0.0),
                ssn_reuse_score: row.get::<f64, _>(5).unwrap_or(0.0),
                age_verification_score: row.get::<f64, _>(6).unwrap_or(0.0),
                address_change_frequency: row.get::<f64, _>(7).unwrap_or(0.0),
                ip_reuse_count: row.get::<f64, _>(8).unwrap_or(0.0),
                employer_verification: row.get::<f64, _>(9).unwrap_or(0.0),
                employment_duration: row.get::<f64, _>(10).unwrap_or(0.0),
                wage_consistency: row.get::<f64, _>(11).unwrap_or(0.0),
                separation_risk: row.get::<f64, _>(12).unwrap_or(0.0),
                ip_address_match: row.get::<f64, _>(13).unwrap_or(0.0),
                multi_state_filing: row.get::<f64, _>(14).unwrap_or(0.0),
                document_quality: row.get::<f64, _>(15).unwrap_or(0.0),
                response_pattern: row.get::<f64, _>(16).unwrap_or(0.0),
                is_fraud: row.get::<f64, _>(17).unwrap_or(0.0),
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
    })
}

#[cfg(not(feature = "database"))]
/// Stub function when database feature is not enabled
pub fn load_from_database(_config: &DatabaseConfig) -> Result<Vec<ClaimRecord>, String> {
    Err("Database feature not enabled. Rebuild with --features database".to_string())
}

/// Export demo data to JSON file for web demonstration
/// 
/// Security: Validates filename to prevent directory traversal attacks
pub fn export_demo_data(records: &[ClaimRecord], filename: &str) -> Result<(), String> {
    // Validate filename to prevent directory traversal
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return Err("Invalid filename: must not contain path separators or '..'".to_string());
    }
    
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
