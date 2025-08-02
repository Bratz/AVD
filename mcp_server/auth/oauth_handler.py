"""
OAuth handler for Banking MCP Server
Based on Christian Posta's approach
"""

import os
import jwt
import httpx
from typing import Dict, Optional, List
from jwt import PyJWKClient
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# Configuration from environment
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
REALM_NAME = "banking-mcp"
JWT_AUDIENCE = "banking-mcp-server"

security = HTTPBearer()

class KeycloakTokenValidator:
    """Validates JWT tokens from Keycloak"""
    
    def __init__(self):
        self.keycloak_url = KEYCLOAK_URL
        self.realm = REALM_NAME
        self.jwt_issuer = f"{self.keycloak_url}/realms/{self.realm}"
        self.jwt_audience = JWT_AUDIENCE
        self.jwks_url = f"{self.jwt_issuer}/protocol/openid-connect/certs"
        self.jwks_client = PyJWKClient(self.jwks_url)
        
    async def validate_token(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(security),
        required_scopes: Optional[List[str]] = None
    ) -> Dict:
        """
        Validate JWT token from Authorization header
        Following Christian Posta's token validation approach
        """
        token = credentials.credentials
        
        try:
            # Get signing key from JWKS
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            
            # Decode and verify token
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.jwt_audience,
                issuer=self.jwt_issuer,
                options={"verify_signature": True, "verify_exp": True}
            )
            
            # Extract scopes from token
            token_scopes = payload.get('scope', '').split()
            
            # Verify required scopes if specified
            if required_scopes:
                missing = set(required_scopes) - set(token_scopes)
                if missing:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Missing required scopes: {', '.join(missing)}"
                    )
            
            # Add user info to payload
            payload['username'] = payload.get('preferred_username', payload.get('sub'))
            payload['scopes'] = token_scopes
            
            logger.info(f"Token validated for user: {payload['username']}")
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidAudienceError:
            raise HTTPException(
                status_code=401, 
                detail=f"Invalid audience, expected {self.jwt_audience}"
            )
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token")

# Global validator instance
token_validator = KeycloakTokenValidator()

def require_scopes(*scopes: str):
    """Dependency to require specific scopes"""
    async def scope_checker(
        payload: Dict = Depends(token_validator.validate_token)
    ):
        token_scopes = payload.get('scopes', [])
        missing = set(scopes) - set(token_scopes)
        if missing:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required scopes: {', '.join(missing)}"
            )
        return payload
    return scope_checker