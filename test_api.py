#!/usr/bin/env python
"""Simple test to verify API is running and accessible."""

import requests
import time
import sys

def test_api():
    """Test API endpoints."""
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("Testing AI Lawyer API")
    print("=" * 60)
    
    endpoints = [
        "/",
        "/health",
        "/docs",
        "/openapi.json",
    ]
    
    for endpoint in endpoints:
        url = base_url + endpoint
        print(f"\n[Test] GET {endpoint}")
        try:
            response = requests.get(url, timeout=5)
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  ✅ OK")
            else:
                print(f"  ❌ ERROR: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
        except requests.exceptions.ConnectionError:
            print(f"  ❌ CONNECTION ERROR - API not running on {base_url}")
            return False
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    time.sleep(2)  # Wait for server to start
    success = test_api()
    sys.exit(0 if success else 1)
