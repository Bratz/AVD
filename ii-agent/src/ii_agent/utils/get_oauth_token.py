# Create scripts/get_oauth_token.py

#!/usr/bin/env python3
"""
Helper script to get OAuth tokens for II-Agent
"""
import asyncio
import sys
sys.path.append('.')

from test_oauth_flow import BankingOAuthTester

async def main():
    print("🔐 OAuth Token Helper for II-Agent")
    print("=" * 50)
    
    tester = BankingOAuthTester()
    
    # Get authorization code
    auth_code = await tester.get_authorization_code_manual()
    if not auth_code:
        print("❌ Failed to get authorization code")
        return
    
    # Exchange for token
    token_data = await tester.exchange_code_for_token(auth_code)
    if not token_data:
        print("❌ Failed to get token")
        return
    
    print("\n✅ OAuth tokens received!")
    print("\nAdd these to your .env file:")
    print(f"OAUTH_TOKEN={token_data['access_token']}")
    print(f"OAUTH_REFRESH_TOKEN={token_data.get('refresh_token', '')}")
    
    print("\nOr export them:")
    print(f"export OAUTH_TOKEN='{token_data['access_token']}'")
    print(f"export OAUTH_REFRESH_TOKEN='{token_data.get('refresh_token', '')}'")

if __name__ == "__main__":
    asyncio.run(main())