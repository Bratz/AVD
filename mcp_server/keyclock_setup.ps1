# install-banking-mcp-windows.ps1
# Banking MCP Phase 1 Windows Installation Script
# Supports both Keycloak and Simplified OAuth options

param(
    [switch]$Simplified,
    [string]$InstallPath = "$env:USERPROFILE\banking-mcp"
)

Write-Host "🚀 Banking MCP Phase 1 - Windows Installation" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Check if running as Administrator for some operations
# $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if ($Simplified) {
    Write-Host "📦 Installing Simplified OAuth Server (Lightweight)" -ForegroundColor Yellow
    $installType = "simplified"
} else {
    Write-Host "📦 Installing Full Keycloak OAuth Server (Enterprise)" -ForegroundColor Yellow
    $installType = "keycloak"
}

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Cyan

# Check Python
try {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion -match "Python 3\.([89]|\d{2})") {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python version too old"
    }
} catch {
    Write-Host "❌ Python 3.8+ is required but not found." -ForegroundColor Red
    Write-Host "📥 Please install Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "   Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    exit 1
}

# Check pip
try {
    pip --version | Out-Null
    Write-Host "✅ pip found" -ForegroundColor Green
} catch {
    Write-Host "❌ pip is required but not found." -ForegroundColor Red
    exit 1
}

# Check Java (only for Keycloak installation)
if ($installType -eq "keycloak") {
    try {
        $javaVersion = java -version 2>&1 | Select-String "version"
        if ($javaVersion -match '1\.[89]|1[1-9]|[2-9][0-9]') {
            Write-Host "✅ Java found: $javaVersion" -ForegroundColor Green
        } else {
            throw "Java version too old"
        }
    } catch {
        Write-Host "❌ Java 11+ is required for Keycloak but not found." -ForegroundColor Red
        Write-Host "📥 Please install Java from:" -ForegroundColor Yellow
        Write-Host "   - Oracle JDK: https://www.oracle.com/java/technologies/downloads/" -ForegroundColor Yellow
        Write-Host "   - OpenJDK: https://adoptium.net/" -ForegroundColor Yellow
        Write-Host "   - Via Chocolatey: choco install openjdk11" -ForegroundColor Yellow
        exit 1
    }
    
    # Check PostgreSQL (for Keycloak)
    try {
        psql --version | Out-Null
        Write-Host "✅ PostgreSQL found" -ForegroundColor Green
    } catch {
        Write-Host "❌ PostgreSQL is required for Keycloak but not found." -ForegroundColor Red
        Write-Host "📥 Please install PostgreSQL from:" -ForegroundColor Yellow
        Write-Host "   - Official: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
        Write-Host "   - Via Chocolatey: choco install postgresql" -ForegroundColor Yellow
        Write-Host "" -ForegroundColor Yellow
        Write-Host "After installing PostgreSQL:" -ForegroundColor Yellow
        Write-Host "1. Start PostgreSQL service" -ForegroundColor Yellow
        Write-Host "2. Remember the password you set for 'postgres' user" -ForegroundColor Yellow
        Write-Host "3. Run this script again" -ForegroundColor Yellow
        exit 1
    }
}

# Create installation directory
Write-Host "📁 Creating installation directory: $InstallPath" -ForegroundColor Cyan
if (!(Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
}
Set-Location $InstallPath

# Create Python virtual environment
Write-Host "🐍 Setting up Python virtual environment..." -ForegroundColor Cyan
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create Python virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Install Python dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Cyan
$requirements = @"
fastapi==0.104.1
uvicorn[standard]==0.24.0
fastmcp==2.2.7
httpx==0.25.2
pyjwt[crypto]==2.8.0
python-dotenv==1.0.0
pyyaml==6.0.1
python-multipart==0.0.6
pydantic==2.5.0
cryptography==41.0.8
authlib==1.2.1
itsdangerous==2.1.2
"@

if ($installType -eq "keycloak") {
    $requirements += "python-keycloak==3.7.0"
}

$requirements | Out-File -FilePath "requirements.txt" -Encoding utf8
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Python dependencies" -ForegroundColor Red
    exit 1
}

if ($installType -eq "keycloak") {
    # Keycloak Installation
    Write-Host "🔐 Setting up Keycloak..." -ForegroundColor Cyan
    
    # Setup PostgreSQL database
    Write-Host "🗄️ Setting up PostgreSQL database for Keycloak..." -ForegroundColor Cyan
    $dbName = "keycloak"
    $dbUser = "keycloak"
    $dbPass = "keycloak_password"
    
    # Create database and user (requires postgres superuser password)
    Write-Host "Please enter the PostgreSQL 'postgres' user password when prompted:" -ForegroundColor Yellow
    
    $createUserSql = @"
CREATE USER $dbUser WITH PASSWORD '$dbPass' CREATEDB;
CREATE DATABASE $dbName OWNER $dbUser;
"@
    
    $createUserSql | psql -U postgres -h localhost
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Keycloak database created" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Database creation may have failed or already exists" -ForegroundColor Yellow
    }
    
    # Download Keycloak
    $keycloakVersion = "23.0.1"
    $keycloakZip = "keycloak-$keycloakVersion.zip"
    $keycloakUrl = "https://github.com/keycloak/keycloak/releases/download/$keycloakVersion/$keycloakZip"
    $keycloakDir = "keycloak-$keycloakVersion"
    
    if (!(Test-Path $keycloakDir)) {
        Write-Host "📥 Downloading Keycloak $keycloakVersion..." -ForegroundColor Cyan
        try {
            Invoke-WebRequest -Uri $keycloakUrl -OutFile $keycloakZip
            Expand-Archive -Path $keycloakZip -DestinationPath . -Force
            Remove-Item $keycloakZip
            Write-Host "✅ Keycloak downloaded and extracted" -ForegroundColor Green
        } catch {
            Write-Host "❌ Failed to download Keycloak: $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "✅ Keycloak already downloaded" -ForegroundColor Green
    }
    
    # Configure Keycloak
    Write-Host "⚙️ Configuring Keycloak..." -ForegroundColor Cyan
    $keycloakConfig = @"
# Database configuration
db=postgres
db-url=jdbc:postgresql://localhost:5432/keycloak
db-username=keycloak
db-password=keycloak_password

# HTTP configuration
http-enabled=true
http-host=0.0.0.0
http-port=8080

# Admin user
admin=admin
admin-password=admin123

# Features
features=dynamic-scopes,client-policies,admin-fine-grained-authz
"@
    
    $keycloakConfig | Out-File -FilePath "$keycloakDir\conf\keycloak.conf" -Encoding utf8
    
    # Create Banking MCP realm configuration
    Write-Host "📝 Creating Banking MCP realm configuration..." -ForegroundColor Cyan
    if (!(Test-Path "keycloak-config")) {
        New-Item -ItemType Directory -Path "keycloak-config" | Out-Null
    }
    
    $realmConfig = @'
{
  "realm": "banking-mcp",
  "displayName": "Banking MCP Realm",
  "enabled": true,
  "registrationAllowed": false,
  "loginWithEmailAllowed": true,
  "duplicateEmailsAllowed": false,
  "resetPasswordAllowed": true,
  "bruteForceProtected": true,
  "accessTokenLifespan": 900,
  "ssoSessionIdleTimeout": 1800,
  "ssoSessionMaxLifespan": 36000,
  "accessCodeLifespan": 60,
  "enabled": true,
  "sslRequired": "external",
  "roles": {
    "realm": [
      {
        "name": "banking_user",
        "description": "Standard banking user role"
      },
      {
        "name": "banking_admin",
        "description": "Banking administrator role"
      }
    ]
  },
  "clientScopes": [
    {
      "name": "banking:read",
      "description": "Read access to banking data",
      "protocol": "openid-connect"
    },
    {
      "name": "banking:write",
      "description": "Write access to banking data",
      "protocol": "openid-connect"
    },
    {
      "name": "customer:profile",
      "description": "Access to customer profile data",
      "protocol": "openid-connect"
    }
  ],
  "clients": [
    {
      "clientId": "banking-mcp-server",
      "name": "Banking MCP Resource Server",
      "enabled": true,
      "clientAuthenticatorType": "client-secret",
      "secret": "mcp-server-secret",
      "standardFlowEnabled": false,
      "serviceAccountsEnabled": true,
      "publicClient": false,
      "protocol": "openid-connect"
    }
  ],
  "users": [
    {
      "username": "banking_user",
      "enabled": true,
      "emailVerified": true,
      "firstName": "Banking",
      "lastName": "User",
      "email": "user@bankingmcp.com",
      "credentials": [
        {
          "type": "password",
          "value": "banking123",
          "temporary": false
        }
      ],
      "realmRoles": ["banking_user"]
    },
    {
      "username": "banking_admin",
      "enabled": true,
      "emailVerified": true,
      "firstName": "Banking",
      "lastName": "Admin",
      "email": "admin@bankingmcp.com",
      "credentials": [
        {
          "type": "password",
          "value": "admin123",
          "temporary": false
        }
      ],
      "realmRoles": ["banking_admin", "banking_user"]
    }
  ]
}
'@
    
    $realmConfig | Out-File -FilePath "keycloak-config\banking-mcp-realm.json" -Encoding utf8
    
} else {
    # Simplified OAuth Installation
    Write-Host "⚡ Setting up Simplified OAuth Server..." -ForegroundColor Cyan
    
    if (!(Test-Path "oauth_server")) {
        New-Item -ItemType Directory -Path "oauth_server" | Out-Null
    }
    
    # Create simplified OAuth server
    $simpleOAuthServer = @'
"""
Simplified OAuth 2.1 Server for Banking MCP - Windows
Lightweight alternative to Keycloak for development/testing
"""

import os
import secrets
import hashlib
import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Form, Query, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import jwt
from pydantic import BaseModel

# Configuration
SECRET_KEY = "your-super-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15

# In-memory storage
clients = {}
authorization_codes = {}
access_tokens = {}
users = {
    "banking_user": {"password": "banking123", "scopes": ["banking:read", "customer:profile"]},
    "banking_admin": {"password": "admin123", "scopes": ["banking:read", "banking:write", "customer:profile"]}
}

app = FastAPI(title="Simple Banking OAuth Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_pkce(code_verifier: str, code_challenge: str) -> bool:
    """Verify PKCE code challenge"""
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip('=')
    return challenge == code_challenge

@app.get("/.well-known/oauth-authorization-server")
async def authorization_server_metadata():
    """OAuth 2.0 Authorization Server Metadata (RFC 8414)"""
    return {
        "issuer": "http://localhost:8080",
        "authorization_endpoint": "http://localhost:8080/oauth/authorize",
        "token_endpoint": "http://localhost:8080/oauth/token",
        "registration_endpoint": "http://localhost:8080/oauth/register",
        "scopes_supported": ["banking:read", "banking:write", "customer:profile"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"]
    }

@app.post("/oauth/register")
async def register_client(request: dict):
    """Dynamic Client Registration (RFC 7591)"""
    client_id = f"mcp_client_{secrets.token_urlsafe(16)}"
    clients[client_id] = {
        "client_id": client_id,
        "redirect_uris": request.get("redirect_uris", []),
        "grant_types": request.get("grant_types", ["authorization_code"]),
        "client_name": request.get("client_name", "MCP Client"),
        "created_at": datetime.utcnow()
    }
    
    return {
        "client_id": client_id,
        "redirect_uris": clients[client_id]["redirect_uris"],
        "grant_types": clients[client_id]["grant_types"],
        "client_name": clients[client_id]["client_name"]
    }

@app.get("/oauth/authorize")
async def authorize(
    response_type: str = Query(...),
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    scope: str = Query(default="banking:read"),
    state: Optional[str] = Query(default=None),
    code_challenge: str = Query(...),
    code_challenge_method: str = Query(default="S256")
):
    """Authorization endpoint with simple auto-approval"""
    
    # Validate client
    if client_id not in clients:
        raise HTTPException(400, "Invalid client_id")
    
    # Auto-approve for demo
    auth_code = secrets.token_urlsafe(32)
    authorization_codes[auth_code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "user_id": "banking_user",
        "expires_at": datetime.utcnow() + timedelta(minutes=10)
    }
    
    # Build redirect URL
    params = f"code={auth_code}"
    if state:
        params += f"&state={state}"
    
    return RedirectResponse(url=f"{redirect_uri}?{params}")

@app.post("/oauth/token")
async def token(
    grant_type: str = Form(...),
    code: str = Form(...),
    redirect_uri: str = Form(...),
    client_id: str = Form(...),
    code_verifier: str = Form(...)
):
    """Token endpoint with PKCE verification"""
    
    if code not in authorization_codes:
        raise HTTPException(400, "Invalid authorization code")
    
    auth_code_data = authorization_codes[code]
    
    if datetime.utcnow() > auth_code_data["expires_at"]:
        raise HTTPException(400, "Authorization code expired")
    
    if not verify_pkce(code_verifier, auth_code_data["code_challenge"]):
        raise HTTPException(400, "Invalid code_verifier")
    
    token_data = {
        "sub": auth_code_data["user_id"],
        "client_id": client_id,
        "scope": auth_code_data["scope"],
        "aud": ["http://localhost:8082"]
    }
    
    access_token = create_access_token(token_data)
    del authorization_codes[code]
    
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "scope": auth_code_data["scope"]
    }

@app.post("/oauth/introspect")
async def introspect_token(token: str = Form(...)):
    """Token introspection endpoint"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "active": True,
            "client_id": payload.get("client_id"),
            "scope": payload.get("scope"),
            "sub": payload.get("sub"),
            "aud": payload.get("aud"),
            "exp": payload.get("exp")
        }
    except jwt.InvalidTokenError:
        return {"active": False}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "simple-oauth-server"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple OAuth Server on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
'@
    
    $simpleOAuthServer | Out-File -FilePath "oauth_server\simple_oauth_server.py" -Encoding utf8
}