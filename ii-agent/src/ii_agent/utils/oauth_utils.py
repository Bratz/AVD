import os
import httpx
import jwt
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OAuthTokenManager:
    """Manages OAuth tokens for II-Agent"""
    
    def __init__(
        self,
        token_endpoint: str,
        client_id: str,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None
    ):
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self._access_token = None
        self._token_expires = None
    
    async def get_valid_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary"""
        # Check if we have a valid token
        if self._access_token and self._token_expires:
            # Add some buffer time (30 seconds) before expiry
            if datetime.now() < (self._token_expires - timedelta(seconds=30)):
                logger.debug(f"Token still valid until {self._token_expires}")
                return self._access_token
            else:
                logger.info("Access token expired or expiring soon, refreshing...")
        
        # Token expired or doesn't exist, try to refresh
        if self.refresh_token:
            try:
                await self.refresh_access_token()
                return self._access_token
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                # Check if refresh token is also expired
                if "invalid_grant" in str(e).lower():
                    logger.error("Refresh token may have expired. Please re-authenticate.")
                return None
        else:
            logger.warning("No refresh token available")
            return None
    
    async def refresh_access_token(self):
        """Refresh the access token using refresh token"""
        logger.info("Attempting to refresh access token...")
        
        async with httpx.AsyncClient() as client:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id
            }
            
            if self.client_secret:
                data["client_secret"] = self.client_secret
            
            try:
                response = await client.post(self.token_endpoint, data=data)
                response.raise_for_status()
                
                token_data = response.json()
                self._access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                
                # Calculate expiry with a smaller buffer (30 seconds instead of 60)
                self._token_expires = datetime.now() + timedelta(seconds=expires_in)
                
                # Update refresh token if provided
                if "refresh_token" in token_data:
                    self.refresh_token = token_data["refresh_token"]
                    logger.info("Refresh token updated")
                
                logger.info(f"Access token refreshed successfully, expires at {self._token_expires}")
                
            except httpx.HTTPStatusError as e:
                error_detail = e.response.text if e.response else "No error details"
                logger.error(f"HTTP error refreshing token: {e.response.status_code} - {error_detail}")
                raise Exception(f"Failed to refresh token: {e.response.status_code} - {error_detail}")
            except Exception as e:
                logger.error(f"Unexpected error refreshing token: {e}")
                raise
    
    @staticmethod
    def from_env() -> Optional['OAuthTokenManager']:
        """Create token manager from environment variables"""
        if os.getenv("ENABLE_OAUTH", "false").lower() != "true":
            return None
        
        keycloak_url = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
        realm = os.getenv("KEYCLOAK_REALM", "banking-mcp")
        token_endpoint = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/token"
        
        refresh_token = os.getenv("OAUTH_REFRESH_TOKEN")
        if not refresh_token:
            logger.warning("OAUTH_REFRESH_TOKEN not found in environment")
            return None
        
        return OAuthTokenManager(
            token_endpoint=token_endpoint,
            client_id=os.getenv("OAUTH_CLIENT_ID", "banking-mcp-client-template"),
            client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
            refresh_token=refresh_token
        )