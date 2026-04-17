"""
Test script for API endpoints
"""

import requests
import json
import time
import subprocess
import threading
import sys

API_URL = "http://127.0.0.1:8000"


def run_api_server():
    """Run the API server in a separate thread"""
    subprocess.Popen([sys.executable, "api/api_server.py"], 
                     cwd="d:/Final Year Project",
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def test_endpoints():
    """Test all API endpoints"""
    # Give server time to start
    time.sleep(5)
    
    print("\n" + "="*50)
    print("Testing API Endpoints")
    print("="*50 + "\n")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"✅ /health: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"❌ /health: {str(e)}\n")
    
    # Test 2: Root endpoint
    try:
        response = requests.get(f"{API_URL}/")
        print(f"✅ / (home): {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"❌ / (home): {str(e)}\n")
    
    # Test 3: Model status
    try:
        response = requests.get(f"{API_URL}/status")
        print(f"✅ /status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"❌ /status: {str(e)}\n")
    
    # Test 4: Prediction endpoint
    try:
        test_data = [0.5, 0.5, 0.5, 0.5, 0.5]
        response = requests.post(f"{API_URL}/predict", json=test_data)
        print(f"✅ /predict: {response.status_code}")
        print(f"   Input: {test_data}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"❌ /predict: {str(e)}\n")
    
    print("="*50)
    print("API Tests Complete!")
    print("="*50)


if __name__ == "__main__":
    print("Starting API test...")
    print("Starting server in background...")
    
    # Run API in background
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Run tests
    test_endpoints()
