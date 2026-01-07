#!/usr/bin/env python3
"""
Web server launcher for FraudNet demo.
Serves from project root so model JSON files are accessible from web/index.html
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import time

PORT = 8000
DEMO_PATH = "/web/"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve files from project root."""
    
    def end_headers(self):
        # Add CORS headers to allow local loading
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Custom logging format
        if args[1] == '200':
            # Only log successful requests to reduce noise
            sys.stdout.write("%s - - [%s] %s\n" %
                           (self.address_string(),
                            self.log_date_time_string(),
                            format % args))

def start_server():
    """Start the HTTP server."""
    # Ensure we're in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Check if model files exist
    model_files = ['model_linear.json', 'model_xor.json', 'model_circular.json']
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if missing_models:
        print(f"⚠️  Warning: Missing model files: {', '.join(missing_models)}")
        print("Run 'cargo run' first to generate model files.")
        return 1
    
    # Check if web directory exists
    if not os.path.exists('web/index.html'):
        print("❌ Error: web/index.html not found!")
        print(f"Current directory: {os.getcwd()}")
        return 1
    
    Handler = CustomHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("\n" + "="*60)
            print("🧠 FraudNet - Client-Side Neural Network Demo")
            print("="*60)
            print(f"🌐 Web server started at http://localhost:{PORT}")
            print(f"📂 Serving from: {os.getcwd()}")
            print(f"🎯 Demo page: http://localhost:{PORT}{DEMO_PATH}")
            print("="*60)
            print("\n✓ Model files found:")
            for model in model_files:
                print(f"  • {model}")
            print(f"\n💡 Opening browser to http://localhost:{PORT}{DEMO_PATH}")
            print("\nPress Ctrl+C to stop the server...\n")
            
            # Open browser after a short delay
            def open_browser():
                time.sleep(1)
                try:
                    # Check if we're in a headless environment
                    import os
                    if os.environ.get('DISPLAY') or os.environ.get('BROWSER'):
                        webbrowser.open(f'http://localhost:{PORT}{DEMO_PATH}')
                        print(f"✓ Browser opened to http://localhost:{PORT}{DEMO_PATH}")
                    else:
                        print(f"⚠️  Running in headless mode - browser not opened")
                        print(f"   Open http://localhost:{PORT}{DEMO_PATH} in your browser")
                except Exception as e:
                    print(f"⚠️  Could not open browser automatically: {e}")
                    print(f"   Please open http://localhost:{PORT}{DEMO_PATH} manually")
            
            threading.Thread(target=open_browser, daemon=True).start()
            
            # Start serving
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
        return 0
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Address already in use
            print(f"\n❌ Error: Port {PORT} is already in use!")
            print(f"Stop the other server or choose a different port.")
            return 1
        else:
            raise

if __name__ == "__main__":
    sys.exit(start_server())
