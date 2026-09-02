from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "I love Dogs"

vector = embeddings.embed_query(text)

print(f"Length of embedding vector: {len(vector)}")
print(f"First 10 elements of embedding vector: {vector[:10]}")