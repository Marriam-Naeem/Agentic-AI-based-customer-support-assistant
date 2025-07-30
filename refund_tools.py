"""
tools/tools.py

Separated tools for the 3-Agent Customer Support System.
Contains individual refund verification and processing tools.
"""

import json
from typing import Dict, Any, Optional
from langchain.tools import tool
from settings import MOCK_REFUND_API


class RefundProcessor:
    """Mock refund processing system for demonstration."""
    
    def __init__(self):
        # Mock database of orders
        self.orders = {
            "12345": {
                "customer_email": "john@example.com",
                "amount": 29.99,
                "product": "Premium Widget",
                "order_date": "2024-01-15",
                "status": "delivered",
                "refund_eligible": True
            },
            "67890": {
                "customer_email": "jane@example.com", 
                "amount": 15.50,
                "product": "Basic Service",
                "order_date": "2024-01-20",
                "status": "delivered",
                "refund_eligible": True
            },
            "11111": {
                "customer_email": "bob@example.com",
                "amount": 99.99,
                "product": "Enterprise Plan",
                "order_date": "2023-12-01",
                "status": "delivered",
                "refund_eligible": False  # Too old
            }
        }
        
        self.processed_refunds = {}
    
    def verify_refund_eligibility(self, order_id: str, customer_email: str = None) -> Dict[str, Any]:
        """
        Verify if a refund request is eligible without processing it.
        
        Args:
            order_id: Order ID to check
            customer_email: Customer email for verification
            
        Returns:
            Dict with verification results
        """
        if order_id not in self.orders:
            return {
                "verified": False,
                "reason": "Order not found in our system",
                "order_id": order_id,
                "action_needed": "Please verify the order number or contact support"
            }
        
        order = self.orders[order_id]
        
        # Check if already refunded
        if order_id in self.processed_refunds:
            return {
                "verified": False,
                "reason": "Refund already processed",
                "order_id": order_id,
                "refund_date": self.processed_refunds[order_id]["processed_date"],
                "action_needed": "No further action needed - refund was already processed"
            }
        
        # Check email match if provided
        if customer_email and order["customer_email"].lower() != customer_email.lower():
            return {
                "verified": False,
                "reason": "Email address does not match our order records",
                "order_id": order_id,
                "action_needed": "Please provide the correct email address used for the order"
            }
        
        # Check eligibility based on business rules
        if not order["refund_eligible"]:
            return {
                "verified": False,
                "reason": "Order is outside our refund policy window",
                "order_id": order_id,
                "order_date": order["order_date"],
                "action_needed": "Contact our support team for special consideration"
            }
        
        # All checks passed
        return {
            "verified": True,
            "order_id": order_id,
            "amount": order["amount"],
            "product": order["product"], 
            "customer_email": order["customer_email"],
            "order_date": order["order_date"],
            "message": "Refund request verified successfully",
            "action_needed": "Ready to process refund"
        }
    
    def execute_refund(self, order_id: str) -> Dict[str, Any]:
        """
        Execute the actual refund processing.
        
        Args:
            order_id: Order ID to refund
            
        Returns:
            Dict with processing results
        """
        if order_id not in self.orders:
            return {
                "success": False,
                "reason": "Order not found in system",
                "order_id": order_id
            }
        
        if order_id in self.processed_refunds:
            return {
                "success": False,
                "reason": "Refund already processed",
                "order_id": order_id,
                "existing_refund_id": self.processed_refunds[order_id]["refund_id"]
            }
        
        order = self.orders[order_id]
        
        # Simulate refund processing
        refund_record = {
            "order_id": order_id,
            "amount": order["amount"],
            "customer_email": order["customer_email"],
            "processed_date": "2024-01-28",  # Mock current date
            "refund_id": f"REF_{order_id}_{len(self.processed_refunds) + 1}",
            "status": "processed",
            "processing_time": "3-5 business days"
        }
        
        self.processed_refunds[order_id] = refund_record
        
        return {
            "success": True,
            "refund_id": refund_record["refund_id"],
            "amount": refund_record["amount"],
            "order_id": order_id,
            "customer_email": refund_record["customer_email"],
            "processed_date": refund_record["processed_date"],
            "processing_time": refund_record["processing_time"],
            "message": f"Refund of ${refund_record['amount']} processed successfully"
        }


# Global refund processor instance
refund_processor = RefundProcessor()


@tool
def refund_verification_tool(order_id: str, customer_email: str = None) -> str:
    """
    Tool for verifying refund eligibility before processing.
    This tool only checks if a refund can be processed but doesn't execute it.
    
    Args:
        order_id: The order ID to verify
        customer_email: Customer email for verification (optional)
        
    Returns:
        JSON string with verification results
    """
    try:
        if not order_id:
            return json.dumps({
                "verified": False,
                "error": "Missing required field: order_id"
            })
        
        result = refund_processor.verify_refund_eligibility(order_id, customer_email)
        return json.dumps(result, indent=2)
    
    except Exception as e:
        return json.dumps({
            "verified": False,
            "error": f"Verification failed: {str(e)}",
            "order_id": order_id,
            "customer_email": customer_email
        }, indent=2)


@tool  
def refund_processing_tool(order_id: str) -> str:
    """
    Tool for executing actual refund processing.
    This tool should only be called after verification is successful.
    
    Args:
        order_id: The order ID to process refund for
        
    Returns:
        JSON string with processing results
    """
    try:
        if not order_id:
            return json.dumps({
                "success": False,
                "error": "Missing required field: order_id"
            })
        
        result = refund_processor.execute_refund(order_id)
        return json.dumps(result, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "order_id": order_id
        }, indent=2)


def get_refund_tools():
    """Return list of refund-related tools for agent binding."""
    return [refund_verification_tool, refund_processing_tool]


def get_verification_tool():
    """Return only the verification tool."""
    return [refund_verification_tool]


def get_processing_tool():
    """Return only the processing tool."""
    return [refund_processing_tool]


# Convenience functions for direct testing
def verify_refund_request(order_id: str, customer_email: str = None) -> dict:
    """Direct function for verifying refund requests (for testing)."""
    result_str = refund_verification_tool.invoke({"order_id": order_id, "customer_email": customer_email})
    return json.loads(result_str)


def process_refund_request(order_id: str) -> dict:
    """Direct function for processing refund requests (for testing)."""
    result_str = refund_processing_tool.invoke({"order_id": order_id})
    return json.loads(result_str)


if __name__ == "__main__":
    # Test the separated tools
    print("Testing Separated Refund Tools:")
    
    print("\n1. Verify valid order:")
    result1 = verify_refund_request("12345", "john@example.com")
    print(json.dumps(result1, indent=2))
    
    print("\n2. Process refund after verification:")
    if result1.get("verified"):
        result2 = process_refund_request("12345")
        print(json.dumps(result2, indent=2))
    
    print("\n3. Verify invalid order:")
    result3 = verify_refund_request("99999", "nobody@example.com")
    print(json.dumps(result3, indent=2))
    
    print("\n4. Try to verify already processed order:")
    result4 = verify_refund_request("12345", "john@example.com")
    print(json.dumps(result4, indent=2))