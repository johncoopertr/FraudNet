use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs;
use std::io::Read;
use tiny_http::{Server, Response, Header, Method};

const PORT: u16 = 8000;
const DEMO_PATH: &str = "/web/";

fn get_content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|s| s.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}

fn serve_file(file_path: &Path) -> Result<Response<std::io::Cursor<Vec<u8>>>, String> {
    let mut file = fs::File::open(file_path)
        .map_err(|e| format!("Failed to open file: {}", e))?;
    
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    
    let content_type = get_content_type(file_path);
    let response = Response::from_data(contents)
        .with_header(Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes())
            .expect("Content-Type header creation failed"))
        .with_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..])
            .expect("CORS header creation failed"))
        .with_header(Header::from_bytes(&b"Access-Control-Allow-Methods"[..], &b"GET"[..])
            .expect("CORS methods header creation failed"))
        .with_header(Header::from_bytes(&b"Cache-Control"[..], &b"no-store, no-cache, must-revalidate"[..])
            .expect("Cache-Control header creation failed"));
    
    Ok(response)
}

fn sanitize_path(url_path: &str) -> Option<PathBuf> {
    // Get the current directory as the base path
    let base_path = std::env::current_dir().ok()?;
    
    // Remove leading slash and convert to path
    let requested_path = url_path.trim_start_matches('/');
    
    // Check for path traversal patterns
    if requested_path.contains("..") {
        return None;
    }
    
    let file_path = base_path.join(requested_path);
    
    // Canonicalize base path
    let canonical_base = base_path.canonicalize().ok()?;
    
    // For existing files, canonicalize and verify they're within base
    if file_path.exists() {
        let canonical_file = file_path.canonicalize().ok()?;
        if canonical_file.starts_with(&canonical_base) {
            return Some(canonical_file);
        }
    } else {
        // For non-existent files, verify the constructed path is within base
        // by checking all components don't escape
        if file_path.starts_with(&base_path) {
            return Some(file_path);
        }
    }
    
    None
}

fn start_server() -> Result<(), Box<dyn std::error::Error>> {
    let model_files = vec!["model_linear.json", "model_xor.json", "model_circular.json"];
    let missing_models: Vec<_> = model_files.iter()
        .filter(|f| !Path::new(f).exists())
        .collect();
    
    if !missing_models.is_empty() {
        println!("⚠️  Warning: Missing model files: {}", 
                 missing_models.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "));
        println!("Run 'cargo run' first to generate model files.");
        return Err("Missing model files".into());
    }
    
    if !Path::new("web/index.html").exists() {
        eprintln!("❌ Error: web/index.html not found!");
        eprintln!("Current directory: {:?}", std::env::current_dir()?);
        return Err("web/index.html not found".into());
    }
    
    let server = Server::http(format!("0.0.0.0:{}", PORT))
        .map_err(|e| {
            if e.to_string().contains("already in use") {
                format!("❌ Error: Port {} is already in use!\nStop the other server or choose a different port.", PORT)
            } else {
                format!("Failed to start server: {}", e)
            }
        })?;
    
    println!("\n{}", "=".repeat(60));
    println!("🧠 FraudNet - Client-Side Neural Network Demo");
    println!("{}", "=".repeat(60));
    println!("🌐 Web server started at http://localhost:{}", PORT);
    println!("📂 Serving from: {:?}", std::env::current_dir()?);
    println!("🎯 Demo page: http://localhost:{}{}", PORT, DEMO_PATH);
    println!("{}", "=".repeat(60));
    println!("\n✓ Model files found:");
    for model in &model_files {
        println!("  • {}", model);
    }
    println!("\n💡 Opening browser to http://localhost:{}{}", PORT, DEMO_PATH);
    println!("\nPress Ctrl+C to stop the server...\n");
    
    // Try to open browser
    let url = format!("http://localhost:{}{}", PORT, DEMO_PATH);
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let _ = open_browser(&url);
    });
    
    // Handle requests
    for request in server.incoming_requests() {
        let url_path = request.url();
        let method = request.method().clone();
        
        // Determine file path with directory traversal protection
        let file_path = if url_path == "/" || url_path == DEMO_PATH {
            sanitize_path("web/index.html")
        } else {
            // Sanitize path to prevent directory traversal
            sanitize_path(url_path)
        };
        
        // Serve file or return 404
        let response = match file_path {
            Some(path) if path.exists() => {
                match serve_file(&path) {
                    Ok(resp) => {
                        // Log successful requests
                        if method == Method::Get {
                            let addr = request.remote_addr()
                                .map(|a| a.to_string())
                                .unwrap_or_else(|| "unknown".to_string());
                            println!("{} - - \"GET {} HTTP/1.1\" 200 -", addr, url_path);
                        }
                        resp
                    }
                    Err(e) => {
                        eprintln!("Error serving file {}: {}", path.display(), e);
                        Response::from_string("500 Internal Server Error").with_status_code(500)
                    }
                }
            }
            _ => Response::from_string("404 Not Found").with_status_code(404)
        };
        
        let _ = request.respond(response);
    }
    
    Ok(())
}

fn open_browser(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(url).spawn()?;
    }
    #[cfg(target_os = "linux")]
    {
        // Check if we're in a headless environment
        if std::env::var("DISPLAY").is_err() {
            println!("⚠️  Running in headless mode - browser may not open");
            println!("   Please open {} manually", url);
            return Ok(());
        }
        Command::new("xdg-open").arg(url).spawn()?;
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd").args(&["/C", "start", url]).spawn()?;
    }
    Ok(())
}

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
    
    match start_server() {
        Ok(_) => {
            println!("\n\n👋 Server stopped. Goodbye!");
        }
        Err(e) => {
            eprintln!("❌ Web server error: {}", e);
            std::process::exit(1);
        }
    }
}
