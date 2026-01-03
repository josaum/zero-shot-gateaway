import requests
import json
import glob
import os
import time

GATEWAY_URL = "http://localhost:9382"
INGEST_URL = f"{GATEWAY_URL}/ingest"
BENCHMARK_DIR = "/Users/josaum/Documents/JAI/multiplan/multiplan"
OUTPUT_DIR = "captured_exports"

def main():
    # 1. Clear previous captures
    for f in glob.glob(f"{OUTPUT_DIR}/*.json"):
        os.remove(f)

    # 2. Ingest 3 representative images
    files = glob.glob(os.path.join(BENCHMARK_DIR, "*.jpg"))
    files = [f for f in files if not os.path.basename(f).startswith("ocr_")]
    files.sort()
    files = files[:3]

    print(f"Triggering ingestion for {len(files)} files to verify extraction quality...")
    
    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"  POST {fname}...")
        requests.post(INGEST_URL, json={
            "type": "InvoiceUploaded",
            "image_path": fpath,
            "message": "Quality Check"
        })

    # 3. Wait for Webhook Capture
    print("Waiting for export (approx 10s due to batching)...")
    for _ in range(20):
        captures = glob.glob(f"{OUTPUT_DIR}/*.json")
        if captures:
            # Check if we have events
            total_events = 0
            for c in captures:
                try:
                    with open(c) as f:
                        data = json.load(f)
                        total_events += data.get("event_count", 0)
                except: pass
            
            if total_events >= len(files):
                print(f"✅ Captured {total_events} events in {len(captures)} files.")
                break
        time.sleep(1)
    
    # 4. Generate Report
    print("\n--- EXTRACTION VERIFICATION ---\n")
    captures = glob.glob(f"{OUTPUT_DIR}/*.json")
    captures.sort()
    
    for c in captures:
        with open(c) as f:
            data = json.load(f)
            abox = data.get("abox_jsonld", {}).get("@graph", [])
            print(f"File: {os.path.basename(c)}")
            print(f"  Event Count: {data.get('event_count')}")
            print(f"  ABox Entities: {len(abox)}")
            
            # Show first few entities
            for i, entity in enumerate(abox[:2]):
                 print(f"    Entity {i}: {json.dumps(entity, indent=0).replace(chr(10), ' ')}")

if __name__ == "__main__":
    main()
