"""
Test script for RunPod NER Distillation Endpoint
"""
import requests
import json
import time

# Endpoint configuration
ENDPOINT_ID = "keer9ba9p4n2tn"
API_URL = f"https://api.runpod.io/v2/{ENDPOINT_ID}/run"
API_KEY = "YOUR_API_KEY_HERE"  # Optional, only if you set one

def test_training_request():
    """Test a training request"""
    print("=" * 60)
    print("🧪 Testing Training Request")
    print("=" * 60)
    
    payload = {
        "input": {
            "action": "train",
            "config": "/app/training_ner/configs/kd_camembert.yaml",
            "output_dir": "/app/artifacts"
        }
    }
    
    print(f"\n📤 Sending request to: {API_URL}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Response: {json.dumps(result, indent=2)}")
            
            if "id" in result:
                job_id = result["id"]
                print(f"\n🎯 Job ID: {job_id}")
                print(f"📍 Check status at: https://console.runpod.io/jobs?id={job_id}")
                return job_id
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {e}")
    
    return None

def test_annotation_request():
    """Test an annotation request"""
    print("\n" + "=" * 60)
    print("🧪 Testing Annotation Request")
    print("=" * 60)
    
    payload = {
        "input": {
            "action": "annotate",
            "corpus_path": "/app/corpus/corpus_test.txt",
            "output_dir": "/app/training_ner/data"
        }
    }
    
    print(f"\n📤 Sending request to: {API_URL}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Response: {json.dumps(result, indent=2)}")
            
            if "id" in result:
                job_id = result["id"]
                print(f"\n🎯 Job ID: {job_id}")
                return job_id
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {e}")
    
    return None

def check_job_status(job_id):
    """Check job status"""
    print(f"\n" + "=" * 60)
    print(f"📊 Checking Job Status: {job_id}")
    print("=" * 60)
    
    status_url = f"https://api.runpod.io/v2/{ENDPOINT_ID}/status/{job_id}"
    
    print(f"\n📤 Status URL: {status_url}")
    
    try:
        response = requests.get(status_url, timeout=10)
        print(f"\n✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Status: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {e}")
    
    return None

def main():
    """Main test function"""
    print("\n" + "🚀" * 30)
    print("RunPod NER Distillation Endpoint Test")
    print("🚀" * 30)
    
    # Test annotation
    print("\n\n1️⃣  Testing Annotation...")
    annotation_job_id = test_annotation_request()
    
    if annotation_job_id:
        print(f"\n⏳ Waiting 3 seconds before checking status...")
        time.sleep(3)
        check_job_status(annotation_job_id)
    
    # Test training
    print("\n\n2️⃣  Testing Training...")
    training_job_id = test_training_request()
    
    if training_job_id:
        print(f"\n⏳ Waiting 3 seconds before checking status...")
        time.sleep(3)
        check_job_status(training_job_id)
    
    print("\n\n" + "✅" * 30)
    print("Test Complete!")
    print("✅" * 30 + "\n")

if __name__ == "__main__":
    main()
