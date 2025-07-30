import sys
import subprocess

def main():
    """Launch the Customer Assistant frontend."""
    try:
        import gradio
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio>=4.0.0"])
    
    try:
        from frontend import main as run_frontend
        run_frontend()
    except ImportError as e:
        print(f"Error importing frontend: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 