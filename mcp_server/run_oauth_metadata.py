#!/usr/bin/env python3
# check_oauth.py - Check if OAuth endpoints are running

import requests
import json
from urllib.parse import urlparse

def check_endpoint(url, description):
    """Check if an endpoint is accessible"""
    print(f"\n🔍 Checking {description}:")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=5)
        print(f"   ✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   📄 Response: {json.dumps(data, indent=2)[:200]}...")
            except:
                print(f"   📄 Response: {response.text[:200]}...")
        elif response.status_code == 404:
            print(f"   ❌ Endpoint not found (404)")
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
            
        return response.status_code
        
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection refused - service not running")
        return None
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout - service not responding")
        return None
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None

def main():
    print("🔐 OAuth Endpoint Checker for Banking MCP")
    print("=" * 50)
    
    # Check if services are running
    services = {
        "MCP Server": "http://localhost:8082",
        "OAuth Proxy": "http://localhost:8081",
        "Keycloak": "http://localhost:8080",
        "II Web Server": "http://localhost:8000"
    }
    
    print("\n📡 Checking services:")
    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            print(f"   ✅ {name}: Running at {url}")
        except:
            try:
                # Try a simple GET
                response = requests.get(url, timeout=2)
                print(f"   ✅ {name}: Running at {url}")
            except:
                print(f"   ❌ {name}: Not running at {url}")
    
    # Check OAuth endpoints
    print("\n🔐 Checking OAuth endpoints:")
    
    # Try both proxy and direct
    endpoints_to_check = [
        ("http://localhost:8081/.well-known/oauth-protected-resource", "OAuth Proxy - Protected Resource"),
        ("http://localhost:8082/.well-known/oauth-protected-resource", "MCP Direct - Protected Resource"),
        ("http://localhost:8080/realms/banking-mcp/.well-known/openid-configuration", "Keycloak - OpenID Config"),
    ]
    
    results = {}
    for url, desc in endpoints_to_check:
        status = check_endpoint(url, desc)
        results[url] = status
    
    # Summary
    print("\n📊 Summary:")
    if results.get("http://localhost:8081/.well-known/oauth-protected-resource") == 200:
        print("   ✅ OAuth proxy is running correctly on port 8081")
        print("   👉 Use http://localhost:8081 with MCP Inspector")
    elif results.get("http://localhost:8082/.well-known/oauth-protected-resource") == 200:
        print("   ✅ OAuth endpoints are served directly by MCP server")
        print("   👉 Use http://localhost:8082 with MCP Inspector")
    else:
        print("   ❌ OAuth endpoints are NOT running")
        print("\n   To fix:")
        print("   1. Run OAuth proxy: python mcp_server/oauth_proxy.py")
        print("   2. OR disable OAuth: set ENABLE_OAUTH=false in .env")
    
    if results.get("http://localhost:8080/realms/banking-mcp/.well-known/openid-configuration") != 200:
        print("\n   ⚠️  Keycloak is not running or realm not configured")
        print("   Start Keycloak with your banking-mcp realm")

if __name__ == "__main__":
    main()