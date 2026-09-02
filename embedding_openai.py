from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

text = "I love Dogs"

vector = openai_embeddings.embed_query(text)

print(f"Length of embedding vector: {len(vector)}")
print(f"First 10 elements of embedding vector: {vector[:10]}")
