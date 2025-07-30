#!/usr/bin/env python3
"""
run_frontend.py

Simple launcher script for the Customer Assistant Gradio frontend.
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import gradio
        print("✅ Gradio is installed")
    except ImportError:
        print("❌ Gradio not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio>=4.0.0"])
        print("✅ Gradio installed successfully")
    
    try:
        import langgraph
        print("✅ LangGraph is installed")
    except ImportError:
        print("❌ LangGraph not found. Please install requirements: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function to launch the frontend."""
    print("🚀 Customer Assistant Frontend Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please install requirements first.")
        sys.exit(1)
    
    # Import and run the frontend
    try:
        from frontend import main as run_frontend
        run_frontend()
    except ImportError as e:
        print(f"❌ Error importing frontend: {e}")
        print("Make sure all required files are present in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 