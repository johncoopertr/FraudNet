use std::process::{Command, Stdio};
use std::path::Path;

fn main() {
    println!("🧠 FraudNet Test-Web");
    println!("====================\n");
    
    // Check if model files exist, if not run cargo run to generate them
    let model_files = vec!["model_linear.json", "model_xor.json", "model_circular.json"];
    let all_exist = model_files.iter().all(|f| Path::new(f).exists());
    
    if !all_exist {
        println!("📝 Model files not found. Training models first...\n");
        
        let status = Command::new("cargo")
            .args(&["run", "--bin", "fraudnet", "--release"])
            .status()
            .expect("Failed to run cargo run");
        
        if !status.success() {
            eprintln!("❌ Failed to train models!");
            std::process::exit(1);
        }
        
        println!("\n✓ Models trained successfully!\n");
    } else {
        println!("✓ Model files found, skipping training.\n");
    }
    
    // Now start the web server
    println!("🌐 Starting web server...\n");
    
    let status = Command::new("python3")
        .arg("scripts/start_webserver.py")
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("Failed to start web server");
    
    if !status.success() {
        eprintln!("❌ Web server exited with error!");
        std::process::exit(1);
    }
}
