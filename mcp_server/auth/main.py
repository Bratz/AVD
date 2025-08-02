# oauth_server/main.py
"""
Banking MCP OAuth 2.1 Authorization Server
Implements RFC 6749, RFC 7636 (PKCE), RFC 7591 (Dynamic Client Registration)
and RFC 8707 (Resource Indicators) for June 2025 MCP compliance
"""

import asyncio
import hashlib
import secrets
import base64
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs, urlencode
from dataclasses import dataclass, asdict
import logging

from fastapi import FastAPI, HTTPException, Depends, Request, Form, Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import jwt
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class OAuthConfig:
    # Server settings
    ISSUER_URL = os.getenv("OAUTH_ISSUER_URL", "https://auth.yourbank.com")
    SERVER_HOST = os.getenv("OAUTH_SERVER_HOST", "localhost")
    SERVER_PORT = int(os.getenv("OAUTH_SERVER_PORT", "8080"))
    
    # JWT settings
    JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 15
    AUTHORIZATION_CODE_EXPIRE_MINUTES = 10
    
    # Supported scopes for banking operations
    SUPPORTED_SCOPES = [
        "banking:read",
        "banking:write", 
        "customer:profile",
        "transaction:history",
        "account:balance"
    ]
    
    # Client registration settings
    ENABLE_DYNAMIC_REGISTRATION = True
    REQUIRE_HTTPS_REDIRECT = False  # Set to True in production

# Data Models
@dataclass
class OAuthClient:
    client_id: str
    client_secret: Optional[str]
    redirect_uris: List[str]
    grant_types: List[str]
    response_types: List[str]
    scope: str
    client_name: Optional[str]
    is_public: bool = True  # MCP clients are typically public
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

@dataclass 
class AuthorizationCode:
    code: str
    client_id: str
    redirect_uri: str
    scope: str
    code_challenge: str
    code_challenge_method: str
    resource_indicator: Optional[str]
    user_id: str
    expires_at: datetime
    used: bool = False

@dataclass
class AccessToken:
    token: str
    client_id: str
    scope: str
    resource_indicator: Optional[str]
    user_id: str
    expires_at: datetime

# Request/Response Models
class ClientRegistrationRequest(BaseModel):
    redirect_uris: List[str]
    grant_types: Optional[List[str]] = ["authorization_code"]
    response_types: Optional[List[str]] = ["code"]
    scope: Optional[str] = "banking:read"
    client_name: Optional[str] = None
    
    @validator('redirect_uris')
    def validate_redirect_uris(cls, v):
        for uri in v:
            parsed = urlparse(uri)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid redirect URI: {uri}")
        return v

class ClientRegistrationResponse(BaseModel):
    client_id: str
    client_secret: Optional[str]
    redirect_uris: List[str]
    grant_types: List[str]
    response_types: List[str]
    scope: str
    client_name: Optional[str]
    client_id_issued_at: int
    client_secret_expires_at: int

class TokenRequest(BaseModel):
    grant_type: str
    code: str
    redirect_uri: str
    client_id: str
    code_verifier: str
    resource: Optional[str] = None  # RFC 8707 Resource Indicator

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: str
    resource: Optional[str] = None  # RFC 8707

# In-memory storage (use database in production)
class MemoryStorage:
    def __init__(self):
        self.clients: Dict[str, OAuthClient] = {}
        self.authorization_codes: Dict[str, AuthorizationCode] = {}
        self.access_tokens: Dict[str, AccessToken] = {}
        
    def store_client(self, client: OAuthClient):
        self.clients[client.client_id] = client
        
    def get_client(self, client_id: str) -> Optional[OAuthClient]:
        return self.clients.get(client_id)
        
    def store_authorization_code(self, auth_code: AuthorizationCode):
        self.authorization_codes[auth_code.code] = auth_code
        
    def get_authorization_code(self, code: str) -> Optional[AuthorizationCode]:
        return self.authorization_codes.get(code)
        
    def use_authorization_code(self, code: str) -> bool:
        if code in self.authorization_codes:
            self.authorization_codes[code].used = True
            return True
        return False
        
    def store_access_token(self, token: AccessToken):
        self.access_tokens[token.token] = token
        
    def get_access_token(self, token: str) -> Optional[AccessToken]:
        return self.access_tokens.get(token)

# OAuth Server Implementation
class BankingOAuthServer:
    def __init__(self):
        self.storage = MemoryStorage()
        self.app = FastAPI(
            title="Banking MCP OAuth 2.1 Authorization Server",
            description="RFC 6749, RFC 7636, RFC 7591, RFC 8707 compliant OAuth server",
            version="1.0.0"
        )
        self.setup_routes()
        self.setup_middleware()
        
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def setup_routes(self):
        # OAuth 2.0 Authorization Server Metadata (RFC 8414)
        @self.app.get("/.well-known/oauth-authorization-server")
        async def authorization_server_metadata():
            return {
                "issuer": OAuthConfig.ISSUER_URL,
                "authorization_endpoint": f"{OAuthConfig.ISSUER_URL}/oauth/authorize",
                "token_endpoint": f"{OAuthConfig.ISSUER_URL}/oauth/token",
                "registration_endpoint": f"{OAuthConfig.ISSUER_URL}/oauth/register",
                "scopes_supported": OAuthConfig.SUPPORTED_SCOPES,
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code"],
                "code_challenge_methods_supported": ["S256"],  # PKCE mandatory
                "token_endpoint_auth_methods_supported": ["none"],  # Public clients
                "resource_parameter_supported": True,  # RFC 8707 Resource Indicators
                "jwks_uri": f"{OAuthConfig.ISSUER_URL}/.well-known/jwks.json"
            }
            
        # Dynamic Client Registration (RFC 7591)
        @self.app.post("/oauth/register", response_model=ClientRegistrationResponse)
        async def register_client(request: ClientRegistrationRequest):
            if not OAuthConfig.ENABLE_DYNAMIC_REGISTRATION:
                raise HTTPException(400, "Dynamic client registration is disabled")
                
            # Generate client credentials
            client_id = f"mcp_client_{secrets.token_urlsafe(16)}"
            client_secret = None  # Public client for MCP
            
            client = OAuthClient(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uris=request.redirect_uris,
                grant_types=request.grant_types,
                response_types=request.response_types,
                scope=request.scope,
                client_name=request.client_name,
                is_public=True
            )
            
            self.storage.store_client(client)
            
            logger.info(f"Registered new MCP client: {client_id}")
            
            return ClientRegistrationResponse(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uris=request.redirect_uris,
                grant_types=request.grant_types,
                response_types=request.response_types,
                scope=request.scope,
                client_name=request.client_name,
                client_id_issued_at=int(client.created_at.timestamp()),
                client_secret_expires_at=0  # Never expires for public clients
            )
            
        # Authorization Endpoint (RFC 6749 + RFC 7636)
        @self.app.get("/oauth/authorize")
        async def authorize(
            response_type: str = Query(...),
            client_id: str = Query(...),
            redirect_uri: str = Query(...),
            scope: str = Query(default="banking:read"),
            state: Optional[str] = Query(default=None),
            code_challenge: str = Query(...),  # PKCE required
            code_challenge_method: str = Query(default="S256"),
            resource: Optional[str] = Query(default=None)  # RFC 8707
        ):
            # Validate client
            client = self.storage.get_client(client_id)
            if not client:
                raise HTTPException(400, "Invalid client_id")
                
            # Validate redirect URI
            if redirect_uri not in client.redirect_uris:
                raise HTTPException(400, "Invalid redirect_uri")
                
            # Validate PKCE parameters
            if code_challenge_method != "S256":
                raise HTTPException(400, "code_challenge_method must be S256")
                
            # For demo purposes, auto-approve (implement user consent in production)
            user_id = "demo_user"  # In production, get from authentication
            
            # Generate authorization code
            auth_code = secrets.token_urlsafe(32)
            code_obj = AuthorizationCode(
                code=auth_code,
                client_id=client_id,
                redirect_uri=redirect_uri,
                scope=scope,
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
                resource_indicator=resource,
                user_id=user_id,
                expires_at=datetime.now(timezone.utc) + timedelta(
                    minutes=OAuthConfig.AUTHORIZATION_CODE_EXPIRE_MINUTES
                )
            )
            
            self.storage.store_authorization_code(code_obj)
            
            # Build redirect URL
            params = {"code": auth_code}
            if state:
                params["state"] = state
                
            redirect_url = f"{redirect_uri}?{urlencode(params)}"
            
            logger.info(f"Authorization granted for client {client_id}, redirecting to {redirect_url}")
            
            return RedirectResponse(url=redirect_url)
            
        # Token Endpoint (RFC 6749 + RFC 7636 + RFC 8707)
        @self.app.post("/oauth/token", response_model=TokenResponse)
        async def token(
            grant_type: str = Form(...),
            code: str = Form(...),
            redirect_uri: str = Form(...),
            client_id: str = Form(...),
            code_verifier: str = Form(...),  # PKCE
            resource: Optional[str] = Form(default=None)  # RFC 8707
        ):
            # Validate grant type
            if grant_type != "authorization_code":
                raise HTTPException(400, "Unsupported grant_type")
                
            # Get authorization code
            auth_code = self.storage.get_authorization_code(code)
            if not auth_code or auth_code.used:
                raise HTTPException(400, "Invalid or expired authorization code")
                
            # Check expiration
            if datetime.now(timezone.utc) > auth_code.expires_at:
                raise HTTPException(400, "Authorization code expired")
                
            # Validate client and redirect URI
            if auth_code.client_id != client_id:
                raise HTTPException(400, "Invalid client_id")
            if auth_code.redirect_uri != redirect_uri:
                raise HTTPException(400, "Invalid redirect_uri")
                
            # Verify PKCE code challenge
            if not self.verify_pkce(code_verifier, auth_code.code_challenge):
                raise HTTPException(400, "Invalid code_verifier")
                
            # Mark code as used
            self.storage.use_authorization_code(code)
            
            # Generate access token
            token_payload = {
                "iss": OAuthConfig.ISSUER_URL,
                "sub": auth_code.user_id,
                "aud": [resource] if resource else [auth_code.resource_indicator] if auth_code.resource_indicator else [],
                "client_id": client_id,
                "scope": auth_code.scope,
                "iat": datetime.now(timezone.utc).timestamp(),
                "exp": (datetime.now(timezone.utc) + timedelta(
                    minutes=OAuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES
                )).timestamp()
            }
            
            access_token = jwt.encode(
                token_payload, 
                OAuthConfig.JWT_SECRET, 
                algorithm=OAuthConfig.JWT_ALGORITHM
            )
            
            # Store token
            token_obj = AccessToken(
                token=access_token,
                client_id=client_id,
                scope=auth_code.scope,
                resource_indicator=resource or auth_code.resource_indicator,
                user_id=auth_code.user_id,
                expires_at=datetime.now(timezone.utc) + timedelta(
                    minutes=OAuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES
                )
            )
            self.storage.store_access_token(token_obj)
            
            logger.info(f"Access token issued for client {client_id}")
            
            response = TokenResponse(
                access_token=access_token,
                expires_in=OAuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                scope=auth_code.scope
            )
            
            # Include resource in response if provided (RFC 8707)
            if resource or auth_code.resource_indicator:
                response.resource = resource or auth_code.resource_indicator
                
            return response
            
        # Token Introspection (for resource servers)
        @self.app.post("/oauth/introspect")
        async def introspect_token(
            token: str = Form(...),
            token_type_hint: Optional[str] = Form(default="access_token")
        ):
            try:
                payload = jwt.decode(
                    token, 
                    OAuthConfig.JWT_SECRET, 
                    algorithms=[OAuthConfig.JWT_ALGORITHM]
                )
                
                return {
                    "active": True,
                    "client_id": payload.get("client_id"),
                    "scope": payload.get("scope"),
                    "sub": payload.get("sub"),
                    "aud": payload.get("aud"),
                    "iss": payload.get("iss"),
                    "exp": payload.get("exp"),
                    "iat": payload.get("iat")
                }
            except jwt.InvalidTokenError:
                return {"active": False}
                
        # JWKS endpoint for token verification
        @self.app.get("/.well-known/jwks.json")
        async def jwks():
            # In production, use proper RSA keys
            return {
                "keys": [{
                    "kty": "oct",
                    "kid": "oauth-key-1",
                    "alg": OAuthConfig.JWT_ALGORITHM,
                    "use": "sig"
                }]
            }
            
        # Health check
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "service": "oauth-server"}
            
    def verify_pkce(self, code_verifier: str, code_challenge: str) -> bool:
        """Verify PKCE code challenge using S256 method"""
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')
        return challenge == code_challenge
        
    def run(self):
        """Run the OAuth server"""
        uvicorn.run(
            self.app,
            host=OAuthConfig.SERVER_HOST,
            port=OAuthConfig.SERVER_PORT,
            log_level="info"
        )

# Main entry point
def main():
    server = BankingOAuthServer()
    logger.info(f"Starting Banking MCP OAuth Server on {OAuthConfig.SERVER_HOST}:{OAuthConfig.SERVER_PORT}")
    logger.info(f"Issuer URL: {OAuthConfig.ISSUER_URL}")
    logger.info(f"Dynamic client registration: {'enabled' if OAuthConfig.ENABLE_DYNAMIC_REGISTRATION else 'disabled'}")
    server.run()

if __name__ == "__main__":
    main()