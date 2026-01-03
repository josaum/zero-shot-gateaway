import requests
import sys

GATEWAY_URL = "http://localhost:9382"
CONFIG_URL = f"{GATEWAY_URL}/api/config"

def configure_webhook(url):
    print(f"Configuring Gateway Webhook to: {url}")
    payload = {
        "webhook_url": url,
        "export_interval_secs": "5", # String expected
        "export_batch_size": "1"     # String expected
    }
    
    try:
        resp = requests.post(CONFIG_URL, json=payload, timeout=5)
        if resp.status_code == 200:
            print("✅ Configuration updated successfully.")
        else:
            print(f"❌ Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9999"
    configure_webhook(target)
