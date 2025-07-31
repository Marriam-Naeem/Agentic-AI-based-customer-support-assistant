import secrets

def generate_api_key(length=32):
    return secrets.token_urlsafe(length)

# Generate an API key of 32 characters
VLLM_API_KEY = generate_api_key()
print("Generated API Key:", VLLM_API_KEY)