from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import sys

OUTPUT_DIR = "captured_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ExportHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            # Save the full export payload
            timestamp = data.get("exported_at", "unknown").replace(":", "-")
            filename = f"{OUTPUT_DIR}/export_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
                
            print(f"✅ Captured export: {filename}")
            
            # Print brief summary
            event_count = data.get("event_count", 0)
            print(f"   Events: {event_count}")
            
            # If ABox present, show sample
            abox = data.get("abox_jsonld", {})
            graph = abox.get("@graph", [])
            print(f"   Entities in ABox: {len(graph)}")

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            
        except Exception as e:
            print(f"❌ Error processing webhook: {e}")
            self.send_response(500)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=ExportHandler, port=9999):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"👂 Webhook Listener running on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Stopped.")

if __name__ == "__main__":
    run()
