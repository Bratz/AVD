import os
import httpx
import logging
from fastmcp import FastMCP
import yaml
from tavily import TavilyClient
from litellm import completion

def register_banking_tools(mcp: FastMCP, config_path: str):
    """Register banking tools with MCP server from a single config file."""
    logger = logging.getLogger(__name__)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    api_key = os.getenv("TCS_BANCS_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    @mcp.tool()
    async def customer_profile(customer_id: str) -> dict:
        """Retrieve customer profile from TCS BaNCS."""
        logger.debug(f"Fetching profile for customer: {customer_id}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://tracsa.sse.com/core/customer/{customer_id}",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return response.json()

    @mcp.tool()
    async def account_balance(customer_id: str) -> dict:
        """Get account balance from TCS BaNCS."""
        logger.debug(f"Fetching balance for customer: {customer_id}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://tracsa.sse.com/core/accounts/{customer_id}",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return response.json()

    @mcp.tool()
    async def transaction(customer_id: str) -> dict:
        """Get transaction history from TCS BaNCS."""
        logger.debug(f"Fetching transactions for customer: {customer_id}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://tracsa.sse.com/core/transactions/{customer_id}",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return response.json()

    @mcp.tool()
    async def market_analysis(query: str) -> dict:
        """Search banking trends via Tavily."""
        logger.debug(f"Searching trends: {query}")
        client = TavilyClient(api_key=tavily_api_key)
        return client.search(query=query, search_depth="advanced")

    @mcp.tool()
    async def financial_advice(prompt: str) -> str:
        """Generate financial advice using LLM."""
        logger.debug(f"Generating advice for prompt: {prompt}")
        response = await completion(
            model=f"ollama/mistral{os.getenv('OLLAMA_MODEL_SUFFIX', ':4bit')}",
            messages=[{"role": "user", "content": prompt}],
            api_base=os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        )
        return response.choices[0].message.content

    logger.info("Banking tools registered")
