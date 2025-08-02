#!/usr/bin/env python3
# enhanced_mcp_diagnostics_v2.py
"""
Enhanced diagnostics for MCP tools registration with OAuth support
Includes fixes for the current OAuth MCP wrapper implementation
"""

import asyncio
import httpx
import jwt
import os
import sys
import yaml
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print("=" * len(text))

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

async def check_service_health():
    """Check if all required services are running"""
    print_header("1️⃣  Service Health Check")
    
    # Get service URLs from environment or use defaults
    keycloak_url = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
    mcp_url = os.getenv("MCP_HTTP_ENDPOINT", "http://localhost:8082")
    
    services = [
        ("Keycloak", f"{keycloak_url}/health/ready", None),
        ("MCP Server", f"{mcp_url.rsplit('/', 1)[0]}/health", None),
        ("MCP Server (root)", f"{mcp_url.rsplit('/', 1)[0]}/", None),
        ("OAuth Metadata", f"{mcp_url.rsplit('/', 1)[0]}/.well-known/oauth-protected-resource", None),
    ]
    
    all_healthy = True
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, url, headers in services:
            try:
                resp = await client.get(url, headers=headers or {})
                if resp.status_code in [200, 204]:
                    print_success(f"{service_name} is running at {url}")
                    
                    # Show service info if available
                    if resp.status_code == 200 and 'json' in resp.headers.get('content-type', ''):
                        try:
                            data = resp.json()
                            if 'oauth_enabled' in data:
                                print_info(f"  OAuth: {'Enabled' if data['oauth_enabled'] else 'Disabled'}")
                            if 'tools_count' in data.get('mcp', {}):
                                print_info(f"  Tools: {data['mcp']['tools_count']}")
                            if 'prompts_count' in data.get('mcp', {}):
                                print_info(f"  Prompts: {data['mcp']['prompts_count']}")
                        except:
                            pass
                else:
                    print_warning(f"{service_name} returned status {resp.status_code}")
                    all_healthy = False
            except httpx.ConnectError:
                print_error(f"{service_name} is not accessible at {url}")
                all_healthy = False
            except Exception as e:
                print_error(f"{service_name} error: {type(e).__name__}: {e}")
                all_healthy = False
    
    return all_healthy

async def check_environment():
    """Check environment variables"""
    print_header("2️⃣  Environment Variables Check")
    
    required_vars = [
        ("KEYCLOAK_URL", "http://localhost:8080"),
        ("KEYCLOAK_REALM", "banking-mcp"),
        ("KEYCLOAK_CLIENT_ID", "banking-mcp-client-template"),
        ("MCP_HTTP_ENDPOINT", "http://localhost:8082/mcp"),
        ("ENABLE_OAUTH", "false"),
    ]
    
    optional_vars = [
        ("MCP_SSE_ENDPOINT", "http://localhost:8084/mcp"),
        ("LOG_LEVEL", "INFO"),
        ("PYTHONPATH", None),
        ("OPENAI_API_KEY", None),
        ("MCP_API_KEY", None),
    ]
    
    missing_required = []
    
    print("\n📋 Required Variables:")
    for var, default in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "SECRET" in var or "PASSWORD" in var or "KEY" in var:
                display_value = value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "****"
            else:
                display_value = value
            print_success(f"{var} = {display_value}")
        else:
            if default:
                print_warning(f"{var} is not set (using default: {default})")
            else:
                print_error(f"{var} is not set")
                missing_required.append(var)
    
    print("\n📋 Optional Variables:")
    for var, default in optional_vars:
        value = os.getenv(var)
        if value:
            if "KEY" in var:
                display_value = value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "****"
            else:
                display_value = value
            print_info(f"{var} = {display_value}")
        else:
            if default:
                print_info(f"{var} is not set (default: {default})")
            else:
                print_warning(f"{var} is not set (optional)")
    
    # Check OAuth status
    oauth_enabled = os.getenv("ENABLE_OAUTH", "false").lower() == "true"
    print(f"\n🔐 OAuth Status: {'ENABLED' if oauth_enabled else 'DISABLED'}")
    
    return len(missing_required) == 0, oauth_enabled

async def get_oauth_token() -> Optional[str]:
    """Get OAuth token from Keycloak"""
    print_header("3️⃣  OAuth Authentication Test")
    
    keycloak_url = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
    realm = os.getenv("KEYCLOAK_REALM", "banking-mcp")
    client_id = os.getenv("KEYCLOAK_CLIENT_ID", "banking-mcp-client-template")
    
    token_url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/token"
    
    print(f"Token URL: {token_url}")
    print(f"Client ID: {client_id}")
    print(f"Username: banking_user")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                token_url,
                data={
                    "grant_type": "password",
                    "client_id": client_id,
                    "username": "banking_user",
                    "password": "banking123",
                    "scope": "openid banking:read banking:write"
                }
            )
            
            if resp.status_code == 200:
                token_data = resp.json()
                token = token_data['access_token']
                
                # Decode token to check claims
                try:
                    decoded = jwt.decode(token, options={"verify_signature": False})
                    print_success("Token acquired successfully")
                    print_info(f"Token expires at: {datetime.fromtimestamp(decoded.get('exp', 0))}")
                    print_info(f"Scopes: {decoded.get('scope', 'N/A')}")
                    print_info(f"Subject: {decoded.get('sub', 'N/A')}")
                    print_info(f"Audience: {decoded.get('aud', 'N/A')}")
                    print_info(f"Authorized party: {decoded.get('azp', 'N/A')}")
                except Exception as e:
                    print_warning(f"Could not decode token: {e}")
                
                return token
            else:
                print_error(f"Failed to get token: {resp.status_code}")
                print_error(f"Response: {resp.text}")
                
                # Common issues
                if resp.status_code == 404:
                    print_info("💡 Check if Keycloak realm 'banking-mcp' exists")
                elif resp.status_code == 401:
                    print_info("💡 Check username/password or client configuration")
                    
                return None
                
        except httpx.ConnectError:
            print_error("Cannot connect to Keycloak")
            print_info("💡 Make sure Keycloak is running at " + keycloak_url)
            return None
        except Exception as e:
            print_error(f"Token request failed: {type(e).__name__}: {e}")
            return None

async def test_mcp_protocol(token: Optional[str], oauth_enabled: bool):
    """Test MCP protocol methods"""
    print_header("4️⃣  MCP Protocol Test")
    
    mcp_endpoint = os.getenv("MCP_HTTP_ENDPOINT", "http://localhost:8082/mcp")
    mcp_url = mcp_endpoint
    
    headers = {"Content-Type": "application/json"}
    if oauth_enabled and token:
        headers["Authorization"] = f"Bearer {token}"
    elif oauth_enabled:
        print_warning("OAuth is enabled but no token available - requests may fail")
    
    # Test initialize
    print("\n🔧 Testing initialize...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "diagnostic-client",
                            "version": "1.0.0"
                        }
                    },
                    "id": 1
                }
            )
            
            if resp.status_code == 200:
                result = resp.json()
                if "result" in result:
                    print_success("Initialize successful")
                    server_info = result["result"]
                    print_info(f"  Protocol version: {server_info.get('protocolVersion', 'N/A')}")
                    print_info(f"  Server: {server_info.get('serverInfo', {}).get('name', 'N/A')} v{server_info.get('serverInfo', {}).get('version', 'N/A')}")
                    
                    caps = server_info.get('capabilities', {})
                    if caps:
                        enabled_caps = [k for k, v in caps.items() if v]
                        print_info(f"  Capabilities: {', '.join(enabled_caps)}")
                else:
                    print_error(f"Initialize failed: {result.get('error', 'Unknown error')}")
            elif resp.status_code == 401:
                print_error("Authentication failed - check OAuth token")
                return []
            else:
                print_error(f"HTTP {resp.status_code}: {resp.text}")
                
        except Exception as e:
            print_error(f"Initialize error: {type(e).__name__}: {e}")
            return []
        
        # Test tools/list
        print("\n🔧 Testing tools/list...")
        try:
            resp = await client.post(
                mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": 2
                }
            )
            
            if resp.status_code == 200:
                result = resp.json()
                if "result" in result:
                    tools = result["result"].get("tools", [])
                    print_success(f"Found {len(tools)} tools")
                    
                    # List all tools
                    if tools:
                        print("\n  📦 Available Tools:")
                        for tool in tools:
                            name = tool.get('name', 'unknown')
                            desc = tool.get('description', 'No description')
                            print(f"    • {name}: {desc[:60]}...")
                    
                    return tools
                else:
                    error = result.get('error', {})
                    print_error(f"tools/list failed: {error.get('message', 'Unknown error')}")
                    if error.get('code') == -32601:
                        print_info("💡 The server might not support the tools/list method")
            else:
                print_error(f"HTTP {resp.status_code}: {resp.text}")
                
        except Exception as e:
            print_error(f"tools/list error: {type(e).__name__}: {e}")
        
        # Test prompts/list
        print("\n🔧 Testing prompts/list...")
        try:
            resp = await client.post(
                mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "prompts/list",
                    "params": {},
                    "id": 3
                }
            )
            
            if resp.status_code == 200:
                result = resp.json()
                if "result" in result:
                    prompts = result["result"].get("prompts", [])
                    print_success(f"Found {len(prompts)} prompts")
                    
                    if prompts:
                        print("\n  💬 Available Prompts:")
                        for prompt in prompts[:5]:  # Show first 5
                            name = prompt.get('name', 'unknown')
                            desc = prompt.get('description', 'No description')
                            print(f"    • {name}: {desc[:60]}...")
                        if len(prompts) > 5:
                            print(f"    ... and {len(prompts) - 5} more")
                
        except Exception as e:
            print_warning(f"prompts/list error: {e}")
    
    return []

async def test_tool_execution(token: Optional[str], tools: List[Dict[str, Any]], oauth_enabled: bool):
    """Test executing specific tools"""
    print_header("5️⃣  Tool Execution Test")
    
    if not tools:
        print_warning("No tools available to test")
        return
    
    mcp_endpoint = os.getenv("MCP_HTTP_ENDPOINT", "http://localhost:8082/mcp")
    headers = {"Content-Type": "application/json"}
    if oauth_enabled and token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Select tools to test - prefer health_check and simple tools
    test_cases = []
    
    # Find health_check tool
    health_tool = next((t for t in tools if t.get('name') == 'health_check'), None)
    if health_tool:
        test_cases.append(("health_check", {}))
    
    # Find list tools
    list_tools = [t for t in tools if 'list' in t.get('name', '').lower()]
    for tool in list_tools[:2]:  # Test up to 2 list tools
        test_cases.append((tool['name'], {}))
    
    if not test_cases:
        # Fallback to first available tool
        test_cases.append((tools[0]['name'], {}))
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for tool_name, args in test_cases:
            print(f"\n🔨 Testing tool: {tool_name}")
            
            try:
                resp = await client.post(
                    mcp_endpoint,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": args
                        },
                        "id": 10 + test_cases.index((tool_name, args))
                    }
                )
                
                if resp.status_code == 200:
                    result = resp.json()
                    if "result" in result:
                        print_success(f"Tool executed successfully")
                        
                        # Pretty print result
                        tool_result = result["result"]
                        if isinstance(tool_result, dict) and "content" in tool_result:
                            content = tool_result["content"]
                            if isinstance(content, list) and content:
                                for item in content[:2]:  # Show first 2 items
                                    if isinstance(item, dict) and "text" in item:
                                        text = item["text"]
                                        if len(text) > 200:
                                            print(f"  Result: {text[:200]}...")
                                        else:
                                            print(f"  Result: {text}")
                        else:
                            print(f"  Result: {str(tool_result)[:200]}...")
                    else:
                        error = result.get('error', {})
                        print_error(f"Tool execution failed: {error.get('message', 'Unknown error')}")
                else:
                    print_error(f"HTTP {resp.status_code}: {resp.text[:200]}...")
                    
            except Exception as e:
                print_error(f"Tool execution error: {type(e).__name__}: {e}")

async def check_oauth_proxy():
    """Check if OAuth proxy is running"""
    print_header("6️⃣  OAuth Proxy Check (Optional)")
    
    proxy_url = os.getenv("OAUTH_PROXY_URL", "http://localhost:8081")
    
    print(f"Checking OAuth proxy at {proxy_url}...")
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            # Check proxy health
            resp = await client.get(f"{proxy_url}/health")
            if resp.status_code == 200:
                print_success("OAuth proxy is running")
                data = resp.json()
                if 'mcp_server' in data:
                    print_info(f"  MCP backend: {data['mcp_server'].get('url', 'N/A')}")
                    print_info(f"  Backend healthy: {data['mcp_server'].get('healthy', False)}")
            else:
                print_warning(f"OAuth proxy returned status {resp.status_code}")
                
        except httpx.ConnectError:
            print_info("OAuth proxy not running (this is optional)")
        except Exception as e:
            print_warning(f"OAuth proxy check failed: {e}")

def generate_report(results: Dict[str, Any]):
    """Generate a summary report with actionable recommendations"""
    print_header("📊 Diagnostic Summary")
    
    issues = []
    recommendations = []
    
    # Analyze results
    if not results.get('services_healthy'):
        issues.append("Some services are not running")
        recommendations.append("Start all required services: Keycloak and MCP Server")
    
    if not results.get('env_vars_ok'):
        issues.append("Missing required environment variables")
        recommendations.append("Copy .env.example to .env and configure required variables")
    
    if results.get('oauth_enabled') and not results.get('token_acquired'):
        issues.append("OAuth is enabled but authentication failed")
        recommendations.append("Check Keycloak configuration and ensure 'banking-mcp' realm exists")
        recommendations.append("Verify user 'banking_user' exists with password 'banking123'")
    
    if results.get('tools_count', 0) == 0:
        issues.append("No tools found in MCP server")
        recommendations.append("Check that tools are being registered in the MCP server")
        recommendations.append("Verify oauth_mcp_wrapper.py is using the fixed discovery method")
    
    # Print results
    if issues:
        print("\n🚨 Issues Found:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print_success("All checks passed! 🎉")
    
    if recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Print quick command reference
    print("\n🔧 Quick Commands:")
    print("```bash")
    print("# Test without OAuth")
    print("export ENABLE_OAUTH=false")
    print("python -m mcp_server.oauth_mcp_wrapper")
    print()
    print("# Test with OAuth")
    print("export ENABLE_OAUTH=true")
    print("python -m mcp_server.oauth_mcp_wrapper")
    print()
    print("# Get OAuth token manually")
    print("curl -X POST http://localhost:8080/realms/banking-mcp/protocol/openid-connect/token \\")
    print("  -d 'grant_type=password&client_id=banking-mcp-client-template&username=banking_user&password=banking123'")
    print("```")
    
    # Success metrics
    print(f"\n📈 Results:")
    print(f"  • Services: {'✓' if results.get('services_healthy') else '✗'}")
    print(f"  • Environment: {'✓' if results.get('env_vars_ok') else '✗'}")
    print(f"  • OAuth: {'✓' if not results.get('oauth_enabled') or results.get('token_acquired') else '✗'}")
    print(f"  • Tools Found: {results.get('tools_count', 0)}")

async def main():
    """Run all diagnostics"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("🏥 MCP Banking System Diagnostics v2.0")
    print("======================================")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.ENDC}")
    
    results = {
        'services_healthy': False,
        'env_vars_ok': False,
        'oauth_enabled': False,
        'token_acquired': False,
        'tools_count': 0,
    }
    
    # Run checks
    results['services_healthy'] = await check_service_health()
    results['env_vars_ok'], results['oauth_enabled'] = await check_environment()
    
    # OAuth test (only if enabled)
    token = None
    if results['oauth_enabled']:
        token = await get_oauth_token()
        results['token_acquired'] = token is not None
    else:
        print_info("OAuth is disabled - skipping authentication test")
    
    # Test MCP protocol
    tools = await test_mcp_protocol(token, results['oauth_enabled'])
    results['tools_count'] = len(tools)
    
    # Test tool execution
    if tools:
        await test_tool_execution(token, tools, results['oauth_enabled'])
    
    # Optional OAuth proxy check
    await check_oauth_proxy()
    
    # Generate report
    generate_report(results)
    
    print(f"\n{Colors.BOLD}Diagnostic completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\n{Colors.FAIL}Fatal error: {type(e).__name__}: {e}{Colors.ENDC}")
        traceback.print_exc()