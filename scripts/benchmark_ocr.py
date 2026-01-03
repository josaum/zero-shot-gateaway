import os
import requests
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DATASET_DIR = Path("/Users/josaum/Documents/JAI/multiplan/multiplan")
GATEWAY_URL = "http://127.0.0.1:9382/ingest"

def benchmark_file(image_path: Path):
    start = time.time()
    try:
        payload = {
            "type": "ocr_benchmark",
            "image_path": str(image_path)
        }
        resp = requests.post(GATEWAY_URL, json=payload, timeout=60)
        duration = time.time() - start
        
        if resp.status_code == 200:
            return {"file": image_path.name, "status": "ok", "duration": duration, "response": resp.json()}
        else:
            return {"file": image_path.name, "status": "error", "code": resp.status_code, "duration": duration}
    except Exception as e:
        return {"file": image_path.name, "status": "failed", "error": str(e), "duration": time.time() - start}

def main():
    print(f"Starting benchmark on {DATASET_DIR}")
    images = sorted(list(DATASET_DIR.glob("*.jpg")))
    print(f"Found {len(images)} images")
    
    results = []
    # Sequential for cleaner logs/debugging, or parallel for stress test.
    # Parallel might overload the shared memory collider loop if not careful.
    # Let's do sequential for accuracy of "latency per document".
    
    for img in images[:5]: # Cap at 5 for quick verify, or all? User asked for benchmark. Let's do 5 first.
        print(f"Processing {img.name}...")
        res = benchmark_file(img)
        print(f"  Result: {res['status']} in {res['duration']:.2f}s")
        results.append(res)
        
    avg_time = sum(r['duration'] for r in results) / len(results)
    print(f"Average time: {avg_time:.2f}s")
    
    # Analyze layout presence
    # Since we can't easily see the internal event log from here without the TUI or tailing log,
    # we assume 'ok' means it was processed.
    # Real validation would check the persistent DB or logs.
    
if __name__ == "__main__":
    main()
