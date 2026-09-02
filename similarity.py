import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

v1 = embeddings.embed_query("I love Dogs")
v2 = embeddings.embed_query("I love Puppies")
v3 = embeddings.embed_query("The cat is on the roof")

print("dogs vs puppies:", cosine_similarity(v1, v2), round(cosine_similarity(v1, v2), 2))
print("dogs vs cats:", cosine_similarity(v1, v3), round(cosine_similarity(v1, v3), 2))