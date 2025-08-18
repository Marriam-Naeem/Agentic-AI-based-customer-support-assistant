from langchain_community.llms import Ollama
aws_ollama = Ollama(
    base_url="http://ec2-65-2-166-41.ap-south-1.compute.amazonaws.com:11434",
    model="llama3:8B",
    temperature=0.2
)
response = aws_ollama.invoke("Hello, how are you?")
print(response)

# Or for chat completion
from langchain_community.chat_models import ChatOllama

chat_ollama = ChatOllama(
    base_url="http://ec2-65-2-166-41.ap-south-1.compute.amazonaws.com:11434",
    model="llama3:8B",
    temperature=0.2
)

chat_response = chat_ollama.invoke("Hello, how are you?")
print(chat_response.content) 