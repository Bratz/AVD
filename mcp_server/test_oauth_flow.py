#!/usr/bin/env python3
# test_oauth_flow_updated.py - Updated OAuth flow test for Banking MCP

import asyncio
import webbrowser
import secrets
import base64
import hashlib
from urllib.parse import urlencode, urlparse, parse_qs
import httpx
import json
from datetime import datetime

class BankingOAuthTester:
    def __init__(self):
        self.keycloak_url = "http://localhost:8080"
        self.realm = "banking-mcp"
        self.client_id = "banking-mcp-client-template"
        self.redirect_uri = "http://localhost:8888/callback"
        self.mcp_server = "http://localhost:8082"
        self.oauth_proxy = "http://localhost:8081"
        
    async def test_complete_flow(self):
        """Test the complete OAuth flow"""
        print("🔐 Banking MCP OAuth Flow Test")
        print("=" * 50)
        
        # Step 1: Check services
        if not await self.check_services():
            return
        
        # Step 2: Get authorization code
        auth_code = await self.get_authorization_code_manual()
        if not auth_code:
            return
            
        # Step 3: Exchange for token
        token_data = await self.exchange_code_for_token(auth_code)
        if not token_data:
            return
            
        # Step 4: Test MCP access
        await self.test_mcp_with_token(token_data['access_token'])
        
        # Step 5: Test token refresh
        if 'refresh_token' in token_data:
            await self.test_token_refresh(token_data['refresh_token'])
        
        print("\n✅ OAuth flow test completed!")
        
    async def check_services(self):
        """Check if all services are running"""
        print("\n1️⃣ Checking services...")
        
        services = {
            "Keycloak": f"{self.keycloak_url}/realms/{self.realm}",
            "MCP Server": f"{self.mcp_server}/health",
            "MCP OAuth Metadata": f"{self.mcp_server}/.well-known/oauth-protected-resource",
        }
        
        # Add OAuth proxy only if configured
        if self.oauth_proxy != self.mcp_server:
            services["OAuth Proxy"] = f"{self.oauth_proxy}/health"
        
        all_ok = True
        async with httpx.AsyncClient() as client:
            for name, url in services.items():
                try:
                    response = await client.get(url, timeout=5)
                    if response.status_code < 400:
                        print(f"   ✅ {name}: Running (status: {response.status_code})")
                        # Show additional info if available
                        if 'json' in response.headers.get('content-type', ''):
                            try:
                                data = response.json()
                                if 'oauth_enabled' in data:
                                    print(f"      OAuth: {'Enabled' if data['oauth_enabled'] else 'Disabled'}")
                                if 'tools_count' in data.get('mcp', {}):
                                    print(f"      Tools: {data['mcp']['tools_count']}")
                            except:
                                pass
                    else:
                        print(f"   ⚠️  {name}: Unhealthy (status: {response.status_code})")
                        all_ok = False
                except Exception as e:
                    print(f"   ❌ {name}: Not running ({type(e).__name__})")
                    if name != "OAuth Proxy":  # OAuth proxy is optional
                        all_ok = False
                    
        return all_ok
        
    async def get_authorization_code_manual(self):
        """Get authorization code using manual browser flow"""
        print("\n2️⃣ Starting authorization flow...")
        
        # Generate PKCE parameters
        self.code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(self.code_verifier.encode()).digest()
        ).decode('utf-8').rstrip('=')
        
        # Store state
        state = secrets.token_urlsafe(16)
        
        # Build auth URL
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid banking:read banking:write customer:profile",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        auth_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/auth?{urlencode(params)}"
        
        print(f"\n📋 Opening browser for login...")
        print(f"   If browser doesn't open, visit:")
        print(f"   {auth_url}\n")
        
        # Start local server to catch callback
        auth_code = await self.start_callback_server()
        
        # Open browser
        webbrowser.open(auth_url)
        
        return auth_code
        
    async def start_callback_server(self):
        """Start a local server to catch the OAuth callback"""
        from aiohttp import web
        
        received_code = None
        
        async def handle_callback(request):
            nonlocal received_code
            code = request.query.get('code')
            state = request.query.get('state')
            error = request.query.get('error')
            
            if error:
                return web.Response(text=f"""
                    <html>
                    <body style="font-family: Arial; text-align: center; padding: 50px;">
                        <h2>❌ Authorization Failed</h2>
                        <p>Error: {error}</p>
                        <p>{request.query.get('error_description', '')}</p>
                    </body>
                    </html>
                """, content_type='text/html')
            
            if code:
                received_code = code
                return web.Response(text="""
                    <html>
                    <body style="font-family: Arial; text-align: center; padding: 50px;">
                        <h2>✅ Authorization Successful!</h2>
                        <p>You can close this window and return to the terminal.</p>
                        <script>setTimeout(() => window.close(), 2000);</script>
                    </body>
                    </html>
                """, content_type='text/html')
            else:
                return web.Response(text="Missing authorization code", status=400)
        
        app = web.Application()
        app.router.add_get('/callback', handle_callback)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8888)
        await site.start()
        
        print("⏳ Waiting for authorization callback...")
        print("   Login with: banking_user / banking123")
        
        # Wait for callback
        for _ in range(120):  # 2 minute timeout
            if received_code:
                break
            await asyncio.sleep(1)
            
        await runner.cleanup()
        
        if received_code:
            print(f"✅ Authorization code received: {received_code[:10]}...")
        else:
            print("❌ Timeout waiting for authorization")
            
        return received_code
        
    async def exchange_code_for_token(self, auth_code):
        """Exchange authorization code for access token"""
        print("\n3️⃣ Exchanging code for token...")
        
        token_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token"
        
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": self.code_verifier
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                print(f"✅ Access token received")
                print(f"   Token type: {token_data.get('token_type')}")
                print(f"   Expires in: {token_data.get('expires_in')} seconds")
                print(f"   Scope: {token_data.get('scope')}")
                
                # Decode token to show claims
                try:
                    import jwt
                    claims = jwt.decode(token_data['access_token'], options={"verify_signature": False})
                    print(f"\n📋 Token claims:")
                    print(f"   Subject: {claims.get('sub')}")
                    print(f"   Username: {claims.get('preferred_username')}")
                    print(f"   Email: {claims.get('email')}")
                    print(f"   Name: {claims.get('name')}")
                    print(f"   Audience: {claims.get('aud')}")
                    print(f"   Scopes: {claims.get('scope')}")
                    
                    # Check for custom claims
                    custom_claims = [k for k in claims.keys() if k not in 
                                   ['sub', 'aud', 'scope', 'exp', 'iat', 'iss', 'jti', 'typ', 
                                    'azp', 'preferred_username', 'email', 'name', 'given_name', 
                                    'family_name', 'email_verified', 'acr', 'realm_access', 
                                    'resource_access', 'allowed-origins', 'session_state', 'sid']]
                    if custom_claims:
                        print(f"   Custom claims: {', '.join(custom_claims)}")
                except ImportError:
                    print("   (Install PyJWT for token decoding: pip install pyjwt)")
                
                return token_data
            else:
                print(f"❌ Token exchange failed: {response.status_code}")
                print(response.text)
                return None
                
    async def test_mcp_with_token(self, access_token):
        """Test MCP access with the OAuth token"""
        print("\n4️⃣ Testing MCP access with token...")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Test 1: Direct MCP server
        print("\n   Testing Direct MCP Server...")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                # Health check
                response = await client.get(
                    f"{self.mcp_server}/health",
                    timeout=10
                )
                print(f"   Health check: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"   OAuth enabled: {data.get('oauth_enabled', False)}")
                    if 'mcp' in data:
                        print(f"   Tools: {data['mcp'].get('tools_count', 0)}")
                        print(f"   Prompts: {data['mcp'].get('prompts_count', 0)}")
                
                # Test MCP initialize
                print("\n   Testing MCP initialize...")
                response = await client.post(
                    f"{self.mcp_server}/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "oauth-test",
                                "version": "1.0.0"
                            }
                        },
                        "id": 1
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"   ✅ MCP initialize successful")
                    result = response.json()
                    if 'result' in result:
                        print(f"   Server: {result['result'].get('serverInfo', {}).get('name')} v{result['result'].get('serverInfo', {}).get('version')}")
                        print(f"   Protocol: {result['result'].get('protocolVersion')}")
                elif response.status_code == 401:
                    print(f"   ❌ Authentication failed - token may be invalid")
                else:
                    print(f"   ❌ MCP initialize failed: {response.status_code}")
                    print(f"   Response: {response.text[:200]}")
                
                # List tools
                print("\n   Testing tools/list...")
                response = await client.post(
                    f"{self.mcp_server}/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/list",
                        "params": {},
                        "id": 2
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'result' in result and 'tools' in result['result']:
                        tools = result['result']['tools']
                        print(f"   ✅ Found {len(tools)} tools")
                        for tool in tools[:5]:  # Show first 5
                            print(f"     - {tool.get('name')}: {tool.get('description', '')[:50]}...")
                        if len(tools) > 5:
                            print(f"     ... and {len(tools) - 5} more")
                            
                        # Try calling health_check tool
                        if any(t['name'] == 'health_check' for t in tools):
                            print("\n   Testing health_check tool...")
                            response = await client.post(
                                f"{self.mcp_server}/mcp",
                                headers=headers,
                                json={
                                    "jsonrpc": "2.0",
                                    "method": "tools/call",
                                    "params": {
                                        "name": "health_check",
                                        "arguments": {}
                                    },
                                    "id": 3
                                },
                                timeout=10
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                if 'result' in result:
                                    print(f"   ✅ health_check tool executed successfully")
                                    content = result['result'].get('content', [])
                                    if content and isinstance(content, list):
                                        print(f"   Result: {content[0].get('text', '')[:100]}...")
                    
            except Exception as e:
                print(f"   ❌ MCP server error: {type(e).__name__}: {str(e)}")
        
        # Test 2: Via OAuth proxy (if different from MCP server)
        if self.oauth_proxy != self.mcp_server:
            print("\n   Testing MCP via OAuth proxy...")
            async with httpx.AsyncClient(follow_redirects=True) as client:
                try:
                    # Check proxy health
                    response = await client.get(f"{self.oauth_proxy}/health", timeout=5)
                    if response.status_code == 200:
                        print(f"   ✅ OAuth proxy is running")
                        
                        # Try listing tools via proxy
                        response = await client.post(
                            f"{self.oauth_proxy}/mcp",
                            headers=headers,
                            json={
                                "jsonrpc": "2.0",
                                "method": "tools/list",
                                "params": {},
                                "id": 4
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            print(f"   ✅ OAuth proxy access successful")
                        else:
                            print(f"   ❌ OAuth proxy access failed: {response.status_code}")
                    else:
                        print(f"   ℹ️  OAuth proxy not available")
                        
                except Exception as e:
                    print(f"   ℹ️  OAuth proxy not configured or running")
                
    async def test_token_refresh(self, refresh_token):
        """Test token refresh flow"""
        print("\n5️⃣ Testing token refresh...")
        
        token_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/token"
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            
            if response.status_code == 200:
                print("   ✅ Token refresh successful")
                new_token = response.json()
                print(f"   New token expires in: {new_token.get('expires_in')} seconds")
            else:
                print(f"   ❌ Token refresh failed: {response.status_code}")

# Quick diagnostic function
async def diagnose_setup():
    """Quick diagnostic of the OAuth setup"""
    print("\n🔍 OAuth Setup Diagnostic")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        # Check Keycloak realm
        try:
            resp = await client.get("http://localhost:8080/realms/banking-mcp")
            print(f"✅ Keycloak realm accessible: {resp.status_code}")
            
            # Check .well-known
            resp = await client.get("http://localhost:8080/realms/banking-mcp/.well-known/openid-configuration")
            config = resp.json()
            print(f"✅ OpenID configuration found")
            print(f"   Authorization endpoint: {config.get('authorization_endpoint')}")
            print(f"   Token endpoint: {config.get('token_endpoint')}")
            print(f"   JWKS URI: {config.get('jwks_uri')}")
            
            # Check if client exists
            print("\n📋 Client Configuration:")
            print(f"   Client ID: banking-mcp-client-template")
            print(f"   Client Type: Public (no secret required)")
            print(f"   PKCE: Required")
            
        except Exception as e:
            print(f"❌ Keycloak check failed: {e}")
        
        # Check MCP endpoints
        print("\n📡 Checking MCP endpoints...")
        endpoints = [
            ("MCP Server", "http://localhost:8082/"),
            ("MCP Health", "http://localhost:8082/health"),
            ("MCP OAuth Metadata", "http://localhost:8082/.well-known/oauth-protected-resource"),
            ("OAuth Proxy (optional)", "http://localhost:8081/health")
        ]
        
        for name, url in endpoints:
            try:
                resp = await client.get(url, timeout=2)
                print(f"   {name}: {resp.status_code} OK")
                if 'oauth_enabled' in resp.text:
                    data = resp.json()
                    if 'oauth_enabled' in data:
                        print(f"     OAuth: {'Enabled' if data['oauth_enabled'] else 'Disabled'}")
            except:
                if "optional" not in name:
                    print(f"   {name}: Not accessible")

# Quick token test
async def test_with_password_grant():
    """Quick test using direct password grant (for testing only)"""
    print("\n🔑 Quick Token Test (Password Grant)")
    print("=" * 50)
    
    token_url = "http://localhost:8080/realms/banking-mcp/protocol/openid-connect/token"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "password",
                "client_id": "banking-mcp-client-template",
                "username": "banking_user",
                "password": "banking123",
                "scope": "openid banking:read banking:write"
            }
        )
        
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Token acquired successfully")
            
            tester = BankingOAuthTester()
            await tester.test_mcp_with_token(token_data['access_token'])
        else:
            print(f"❌ Failed to get token: {response.status_code}")
            print(response.text)

# Main entry point
async def main():
    print("Banking MCP OAuth Test Suite")
    print("============================")
    print("1. Full OAuth flow (with browser)")
    print("2. Quick diagnostics")
    print("3. Test with existing token")
    print("4. Quick test (password grant)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        tester = BankingOAuthTester()
        await tester.test_complete_flow()
    elif choice == "2":
        await diagnose_setup()
    elif choice == "3":
        token = input("Enter your access token: ").strip()
        tester = BankingOAuthTester()
        await tester.test_mcp_with_token(token)
    elif choice == "4":
        await test_with_password_grant()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    # Install required packages if needed
    required = ["aiohttp", "httpx", "pyjwt"]
    import importlib
    import subprocess
    
    for package in required:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call(["pip", "install", package])
    
    asyncio.run(main())