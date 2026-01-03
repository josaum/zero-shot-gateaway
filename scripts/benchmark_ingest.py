import requests
import os
import time
import json
import glob
import sys
from pathlib import Path

# Add scripts directory to path to import consumer
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from consumer import PhysicsConsumer
except ImportError:
    print("❌ Could not import consumer.py. Make sure it is in the same directory.")
    sys.exit(1)

GATEWAY_URL = "http://localhost:9382"
INGEST_URL = f"{GATEWAY_URL}/ingest"
BENCHMARK_DIR = "/Users/josaum/Documents/JAI/multiplan/multiplan"

def main():
    # Find all JPGs
    files = glob.glob(os.path.join(BENCHMARK_DIR, "*.jpg"))
    files = [f for f in files if not os.path.basename(f).startswith("ocr_")]
    files.sort()

    print(f"Found {len(files)} benchmark images.")
    
    # Process a batch
    BATCH_SIZE = 5
    files = files[:BATCH_SIZE]
    
    if len(files) == 0:
        print("No files found!")
        return

    print(f"Processing {len(files)} images...")

    try:
        consumer = PhysicsConsumer()
        consumer.open()
        
        # Get baseline frame count from shared memory header
        initial_head = consumer.header.head
        initial_frame_id = -1
        
        latest = consumer.read_latest()
        if latest:
            initial_frame_id = latest.frame_id
            
        print(f"Initial SHM Head: {initial_head}, Latest Frame ID: {initial_frame_id}")

        start_time = time.time()

        # 1. Fire Ingestion Requests
        for i, fpath in enumerate(files):
            print(f"[{i+1}/{len(files)}] POSTing {os.path.basename(fpath)}...")
            
            payload = {
                "type": "InvoiceUploaded",
                "image_path": fpath,
                "message": "Benchmark"
            }
            
            try:
                requests.post(INGEST_URL, json=payload, timeout=30)
            except Exception as e:
                 print(f"  ⚠️ Error: {e}")

        # 2. Wait for Processing in Shared Memory
        # We expect 'len(files)' new frames to appear
        target_frame_count = len(files)
        frames_seen = 0
        last_seen_id = initial_frame_id
        
        print(f"Monitoring Shared Memory for {target_frame_count} new frames...")

        timeout = 60 # seconds
        wait_start = time.time()
        
        while frames_seen < target_frame_count:
            if time.time() - wait_start > timeout:
                print("❌ Timeout waiting for frames!")
                break
                
            # Polling strategy: check latest frame
            latest = consumer.read_latest()
            
            if latest and latest.frame_id > last_seen_id:
                # We found a new frame!
                # Note: In high throughput, we might jump multiple IDs if we poll slowly.
                # But read_latest() just gives us the tip.
                # Ideally we track the head movement.
                
                # Check how much head moved
                # A robust consumer tracks via follow(), but for bench we just need to know when we are done.
                
                # Simple approach: Wait until last_seen_id >= initial + count ??
                # Frame IDs are monotonic.
                
                diff = latest.frame_id - last_seen_id
                frames_seen += diff
                last_seen_id = latest.frame_id
                print(f"  Processed frame {latest.frame_id} (Total: {frames_seen}/{target_frame_count})")
            
            time.sleep(0.01) # 10ms poll

        total_time = time.time() - start_time
        throughput = frames_seen / total_time
        
        # Calculate Latency stats if possible?
        # The 'elapsed_ms' in the frame is the processing time of that frame!
        
        print("\n" + "="*40)
        print(f"ZERO-COPY BENCHMARK RESULTS")
        print("="*40)
        print(f"Processed:       {frames_seen} / {len(files)}")
        print(f"Total Time:      {total_time:.2f} s")
        print(f"Throughput:      {throughput:.2f} invoices/sec")
        
        if latest:
             print(f"Last Frame Self-Reported Duration: {latest.duration_ms:.2f} ms")
        print("="*40)

        consumer.close()

    except FileNotFoundError:
        print("❌ Shared Memory not found! Is the Gateway running?")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
