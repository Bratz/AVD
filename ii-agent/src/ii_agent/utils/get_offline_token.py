#!/usr/bin/env python3
"""Get OAuth tokens with offline access (30-day validity)"""
import asyncio
import sys
sys.path.append('.')

from test_oauth_flow import BankingOAuthTester

class OfflineOAuthTester(BankingOAuthTester):
    async def get_authorization_code_manual(self):
        """Override to request offline_access scope"""
        import webbrowser
        import secrets
        import base64
        import hashlib
        from urllib.parse import urlencode
        from aiohttp import web
        
        print("\n🔑 Requesting OFFLINE tokens (30-day validity)...")
        
        # Generate PKCE
        self.code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(self.code_verifier.encode()).digest()
        ).decode('utf-8').rstrip('=')
        
        state = secrets.token_urlsafe(16)
        
        # IMPORTANT: Added offline_access scope
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid banking:read banking:write customer:profile offline_access",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        auth_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/auth?{urlencode(params)}"
        
        print(f"\n🌐 Opening browser: {auth_url}")
        webbrowser.open(auth_url)
        
        # Setup callback server
        received_code = None
        
        async def callback_handler(request):
            nonlocal received_code
            code = request.query.get('code')
            if code:
                received_code = code
                return web.Response(text="✅ Success! Close this window.", content_type='text/html')
            return web.Response(text="❌ Failed!", content_type='text/html')
        
        app = web.Application()
        app.router.add_get('/callback', callback_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8888)
        await site.start()
        
        print("⏳ Waiting for authorization...")
        
        for _ in range(120):
            if received_code:
                break
            await asyncio.sleep(1)
            
        await runner.cleanup()
        return received_code

async def main():
    tester = OfflineOAuthTester()
    
    # Get auth code
    auth_code = await tester.get_authorization_code_manual()
    if not auth_code:
        print("❌ Failed to get authorization")
        return
    
    # Exchange for tokens
    token_data = await tester.exchange_code_for_token(auth_code)
    if not token_data:
        print("❌ Failed to get tokens")
        return
    
    print("\n✅ Offline tokens received!")
    print("\n📝 Update your .env file:")
    print(f"OAUTH_TOKEN={token_data['access_token']}")
    print(f"OAUTH_REFRESH_TOKEN={token_data['refresh_token']}")
    print("\n✨ These tokens will work for 30 DAYS!")

if __name__ == "__main__":
    asyncio.run(main())