"""
TCS BaNCS Banking Tools - Generated from OpenAPI Specification
Generated using openapi_to_mcp.py

This file contains FastMCP tools for TCS BaNCS banking operations.
"""

import os
import httpx
import logging
from fastmcp import FastMCP
from typing import Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)

# TCS BaNCS API Configuration
BASE_URL = "https://demoapps.tcsbancs.com/Core"
API_KEY = os.getenv("TCS_BANCS_API_KEY")
TIMEOUT = int(os.getenv("TCS_BANCS_TIMEOUT", "30"))


def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for TCS BaNCS API."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
        # Alternative auth methods based on TCS BaNCS requirements
        # headers["X-API-Key"] = API_KEY
        # headers["TCS-Auth-Token"] = API_KEY
    
    return headers


async def make_bancs_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Make authenticated request to TCS BaNCS API."""
    url = f"{BASE_URL.rstrip('/')}{path}"
    
    
    headers = get_auth_headers()
    
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            logger.debug(f"TCS BaNCS {method} {url} with params: {params}")
            
            if method.upper() == "GET":
                response = await client.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, json=data, params=params, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data, params=params, headers=headers)
            elif method.upper() == "PATCH":
                response = await client.patch(url, json=data, params=params, headers=headers)
            elif method.upper() == "DELETE":
                response = await client.delete(url, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Handle different response types
            content_type = response.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                return response.json()
            else:
                return {"data": response.text, "content_type": content_type}
                
        except httpx.HTTPStatusError as e:
            logger.error(f"TCS BaNCS API error {e.response.status_code}: {e.response.text}")
            error_detail = "Unknown error"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("message", error_data.get("error", str(error_data)))
            except:
                error_detail = e.response.text or str(e)
            
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": error_detail
            }
        except Exception as e:
            logger.error(f"TCS BaNCS API request failed: {str(e)}")
            return {
                "error": True,
                "message": str(e)
            }

def register_bancs_tools(mcp: FastMCP):
    """Register all TCS BaNCS tools with FastMCP server."""
    
    
    @mcp.tool()
    async def get_account_block_list(accountReference: Any, pageNum: Any = None, pageSize: Any = None
    ) -> Dict[str, Any]:
        """Get Account Block List
        
        Generated from: GET /accountManagement/account/blockList/{accountReference}
        Tags: Account Management
        Authentication: Required
        """
        
        # Prepare query parameters
        params = {}
        if accountReference is not None:
            params["accountReference"] = accountReference
        if pageNum is not None:
            params["pageNum"] = pageNum
        if pageSize is not None:
            params["pageSize"] = pageSize
        
        
        # Replace path parameters in URL
        path = "/accountManagement/account/blockList/{accountReference}"
        if "{accountReference}" in path and accountReference is not None:
            path = path.replace("{accountReference}", str(accountReference))
            params.pop("accountReference", None)
        if "{pageNum}" in path and pageNum is not None:
            path = path.replace("{pageNum}", str(pageNum))
            params.pop("pageNum", None)
        if "{pageSize}" in path and pageSize is not None:
            path = path.replace("{pageSize}", str(pageSize))
            params.pop("pageSize", None)
        
        
        return await make_bancs_request("GET", path, params=params)
    
    
    
    @mcp.tool()
    async def get_account_balance(accountReference: Any
    ) -> Dict[str, Any]:
        """Get Account Balance
        
        Generated from: GET /accountManagement/account/balanceDetails/{accountReference}
        Tags: Account Management
        Authentication: Required
        """
        
        # Prepare query parameters
        params = {}
        if accountReference is not None:
            params["accountReference"] = accountReference
        
        
        # Replace path parameters in URL
        path = "/accountManagement/account/balanceDetails/{accountReference}"
        if "{accountReference}" in path and accountReference is not None:
            path = path.replace("{accountReference}", str(accountReference))
            params.pop("accountReference", None)
        
        
        return await make_bancs_request("GET", path, params=params)
    
    
    
    @mcp.tool()
    async def get_loan_details(loanReference: Any
    ) -> Dict[str, Any]:
        """Get Loan Details
        
        Generated from: GET /loans/loan/detailsIB/{loanReference}
        Tags: Loan Management
        Authentication: Required
        """
        
        # Prepare query parameters
        params = {}
        if loanReference is not None:
            params["loanReference"] = loanReference
        
        
        # Replace path parameters in URL
        path = "/loans/loan/detailsIB/{loanReference}"
        if "{loanReference}" in path and loanReference is not None:
            path = path.replace("{loanReference}", str(loanReference))
            params.pop("loanReference", None)
        
        
        return await make_bancs_request("GET", path, params=params)
    
    
    
    @mcp.tool()
    async def create_customer(
    ) -> Dict[str, Any]:
        """Create Customer
        
        Generated from: POST /customerManagement/customer
        Tags: Customer Management
        Authentication: Required
        """
        
        # Prepare request data
        data = {}
        query_params = {}
        
        
        
        # Replace path parameters in URL
        path = "/customerManagement/customer"
        
        
        return await make_bancs_request("POST", path, data=data, params=query_params)
    
    
    
    @mcp.tool()
    async def get_customer_details(customerId: Any
    ) -> Dict[str, Any]:
        """Get Customer Details
        
        Generated from: GET /customerManagement/customer/{customerId}
        Tags: Customer Management
        Authentication: Required
        """
        
        # Prepare query parameters
        params = {}
        if customerId is not None:
            params["customerId"] = customerId
        
        
        # Replace path parameters in URL
        path = "/customerManagement/customer/{customerId}"
        if "{customerId}" in path and customerId is not None:
            path = path.replace("{customerId}", str(customerId))
            params.pop("customerId", None)
        
        
        return await make_bancs_request("GET", path, params=params)
    
    
    
    @mcp.tool()
    async def create_booking(
    ) -> Dict[str, Any]:
        """Create Booking
        
        Generated from: POST /financialAccounting/CreateBooking
        Tags: Financial Accounting
        Authentication: Required
        """
        
        # Prepare request data
        data = {}
        query_params = {}
        
        
        
        # Replace path parameters in URL
        path = "/financialAccounting/CreateBooking"
        
        
        return await make_bancs_request("POST", path, data=data, params=query_params)
    
    
    
    @mcp.tool()
    async def initiate_payment(
    ) -> Dict[str, Any]:
        """Initiate Payment
        
        Generated from: POST /payments/initiate
        Tags: Payments
        Authentication: Required
        """
        
        # Prepare request data
        data = {}
        query_params = {}
        
        
        
        # Replace path parameters in URL
        path = "/payments/initiate"
        
        
        return await make_bancs_request("POST", path, data=data, params=query_params)
    
    
    
    @mcp.tool()
    async def get_transaction_status(transactionId: Any
    ) -> Dict[str, Any]:
        """Get Transaction Status
        
        Generated from: GET /payments/transaction/{transactionId}
        Tags: Payments
        Authentication: Required
        """
        
        # Prepare query parameters
        params = {}
        if transactionId is not None:
            params["transactionId"] = transactionId
        
        
        # Replace path parameters in URL
        path = "/payments/transaction/{transactionId}"
        if "{transactionId}" in path and transactionId is not None:
            path = path.replace("{transactionId}", str(transactionId))
            params.pop("transactionId", None)
        
        
        return await make_bancs_request("GET", path, params=params)
    
    
    
    @mcp.tool()
    async def perform_amlcheck(
    ) -> Dict[str, Any]:
        """Perform AML Check
        
        Generated from: POST /compliance/amlCheck
        Tags: Compliance
        Authentication: Required
        """
        
        # Prepare request data
        data = {}
        query_params = {}
        
        
        
        # Replace path parameters in URL
        path = "/compliance/amlCheck"
        
        
        return await make_bancs_request("POST", path, data=data, params=query_params)
    
    
    
    @mcp.tool()
    async def create_account(
    ) -> Dict[str, Any]:
        """Create Account
        
        Generated from: POST /accountManagement/account
        Tags: Account Management
        Authentication: Required
        """
        
        # Prepare request data
        data = {}
        query_params = {}
        
        
        
        # Replace path parameters in URL
        path = "/accountManagement/account"
        
        
        return await make_bancs_request("POST", path, data=data, params=query_params)
    
    
    
    @mcp.tool()
    async def post_transaction(
    ) -> Dict[str, Any]:
        """Post Transaction
        
        Generated from: POST /transactions/post
        Tags: Transactions
        Authentication: Required
        """
        
        # Prepare request data
        data = {}
        query_params = {}
        
        
        
        # Replace path parameters in URL
        path = "/transactions/post"
        
        
        return await make_bancs_request("POST", path, data=data, params=query_params)
    
    
    
    logger.info("Registered 11 TCS BaNCS tools")

# For backward compatibility with existing banking_tools.py
def register_banking_tools(mcp: FastMCP, config_path: str = None):
    """Legacy function name for compatibility."""
    register_bancs_tools(mcp)