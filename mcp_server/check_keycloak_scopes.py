#!/usr/bin/env python3
# check_keycloak_scopes.py
"""
Check what scopes are actually available in Keycloak
"""

import httpx
import asyncio
import json

KEYCLOAK_URL = "http://localhost:8080"
REALM = "banking-mcp"

async def check_well_known():
    """Check .well-known configuration"""
    async with httpx.AsyncClient() as client:
        # Get OpenID configuration
        resp = await client.get(
            f"{KEYCLOAK_URL}/realms/{REALM}/.well-known/openid-configuration"
        )
        
        if resp.status_code == 200:
            config = resp.json()
            print("🔍 OpenID Configuration:")
            print(f"   Issuer: {config.get('issuer')}")
            print(f"   Token endpoint: {config.get('token_endpoint')}")
            
            # Check scopes_supported
            if 'scopes_supported' in config:
                print(f"\n📋 Supported scopes:")
                for scope in config['scopes_supported']:
                    print(f"   - {scope}")
            else:
                print("\n⚠️  No scopes_supported field in configuration")
                
            return config
        else:
            print(f"❌ Failed to get .well-known configuration: {resp.status_code}")

async def test_minimal_token():
    """Test getting a token with minimal scopes"""
    async with httpx.AsyncClient() as client:
        print("\n🔐 Testing minimal token request (no scopes)...")
        resp = await client.post(
            f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "banking-mcp-client-template",
                "username": "banking_user",
                "password": "banking123"
                # No scope parameter
            }
        )
        
        if resp.status_code == 200:
            token_data = resp.json()
            print("✅ Token obtained without scope parameter")
            print(f"   Default scopes: {token_data.get('scope')}")
            return True
        else:
            print(f"❌ Failed: {resp.status_code}")
            print(f"   Error: {resp.json()}")
            return False

async def test_each_scope():
    """Test each scope individually"""
    scopes_to_test = [
        "openid",
        "profile", 
        "email",
        "banking:read",
        "banking:write",
        "customer:profile",
        "offline_access"
    ]
    
    async with httpx.AsyncClient() as client:
        print("\n🧪 Testing each scope individually:")
        
        working_scopes = []
        for scope in scopes_to_test:
            resp = await client.post(
                f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token",
                data={
                    "grant_type": "password",
                    "client_id": "banking-mcp-client-template",
                    "username": "banking_user",
                    "password": "banking123",
                    "scope": scope
                }
            )
            
            if resp.status_code == 200:
                print(f"   ✅ {scope}: OK")
                working_scopes.append(scope)
            else:
                error = resp.json()
                print(f"   ❌ {scope}: {error.get('error_description', 'Failed')}")
        
        # Test working scopes together
        if working_scopes:
            print(f"\n🔗 Testing working scopes together: {' '.join(working_scopes)}")
            resp = await client.post(
                f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token",
                data={
                    "grant_type": "password",
                    "client_id": "banking-mcp-client-template",
                    "username": "banking_user",
                    "password": "banking123",
                    "scope": " ".join(working_scopes)
                }
            )
            
            if resp.status_code == 200:
                print("✅ Combined working scopes successful")
            else:
                print("❌ Combined scopes failed")

async def main():
    print("🔍 Checking Keycloak Scope Configuration")
    print("=" * 50)
    
    # Check well-known configuration
    config = await check_well_known()
    
    # Test minimal token
    await test_minimal_token()
    
    # Test each scope
    await test_each_scope()
    
    print("\n💡 Recommendations:")
    print("1. If custom scopes fail, they may not be properly configured")
    print("2. Check Keycloak Admin Console → Client Scopes")
    print("3. Ensure scopes are added to client's Optional Client Scopes")
    print("4. Check that 'include.in.token.scope' attribute is set to 'true'")

if __name__ == "__main__":
    asyncio.run(main())