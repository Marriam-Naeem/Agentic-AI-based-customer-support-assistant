"""
tools/refund_tools.py

Refund processing tools for the 3-Agent Customer Support System.
"""

import json
import os
from typing import Dict, Any
from langchain.tools import tool


class RefundProcessor:
    """Refund processing system using JSON database."""
    
    def __init__(self, db_path: str = "./data/orders_database.json"):
        self.db_path = db_path
        self._load_db()
    
    def _load_db(self):
        """Load database from JSON file."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.orders = data.get("orders", {})
                    self.refunds = data.get("processed_refunds", {})
            else:
                self.orders = self.refunds = {}
        except:
            self.orders = self.refunds = {}
    
    def _save_db(self):
        """Save database to JSON file."""
        try:
            data = {
                "orders": self.orders,
                "processed_refunds": self.refunds
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def verify_refund_eligibility(self, order_id: str, customer_email: str = None) -> Dict[str, Any]:
        """Verify if a refund request is eligible."""
        if order_id not in self.orders:
            return {"verified": False, "reason": "Order not found", "order_id": order_id}
        
        order = self.orders[order_id]
        
        if order_id in self.refunds:
            return {"verified": False, "reason": "Refund already processed", "order_id": order_id}
        
        if customer_email and order["customer_email"].lower() != customer_email.lower():
            return {"verified": False, "reason": "Email mismatch", "order_id": order_id}
        
        if not order["refund_eligible"]:
            return {"verified": False, "reason": "Outside refund window", "order_id": order_id}
        
        return {
            "verified": True, "order_id": order_id, "amount": order["amount"],
            "product": order["product"], "customer_email": order["customer_email"],
            "message": "Refund request verified successfully"
        }
    
    def execute_refund(self, order_id: str) -> Dict[str, Any]:
        """Execute the actual refund processing."""
        if order_id not in self.orders or order_id in self.refunds:
            return {"success": False, "reason": "Invalid order or already refunded", "order_id": order_id}
        
        order = self.orders[order_id]
        refund_id = f"REF_{order_id}_{len(self.refunds) + 1}"
        
        self.refunds[order_id] = {
            "order_id": order_id, "amount": order["amount"],
            "customer_email": order["customer_email"], "refund_id": refund_id,
            "processed_date": "2024-01-28", "status": "processed"
        }
        
        # Save the updated database
        self._save_db()
        
        return {
            "success": True, "refund_id": refund_id, "amount": order["amount"],
            "order_id": order_id, "message": f"Refund of ${order['amount']} processed successfully"
        }


refund_processor = RefundProcessor()


@tool
def refund_verification_tool(order_id: str, customer_email: str = None) -> str:
    """Verify refund eligibility before processing."""
    if not order_id:
        return json.dumps({"verified": False, "error": "Missing order_id"})
    
    try:
        result = refund_processor.verify_refund_eligibility(order_id, customer_email)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"verified": False, "error": str(e)}, indent=2)


@tool  
def refund_processing_tool(order_id: str) -> str:
    """Execute actual refund processing after verification."""
    if not order_id:
        return json.dumps({"success": False, "error": "Missing order_id"})
    
    try:
        result = refund_processor.execute_refund(order_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def get_refund_tools():
    """Return list of refund-related tools for agent binding."""
    return [refund_verification_tool, refund_processing_tool]