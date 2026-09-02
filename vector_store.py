from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


text = ["I love my buji and scheru", "They are like my children", "I love programming in Python", "LangChain is a great framework for building applications with LLMs"]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(text, embeddings)

print("Vector store created with the following texts:")
for t in text:
    print(t)

query = "Tell me about your pets"
results = vector_store.similarity_search(query, k=2) # Retrieve the top 2 most similar texts to the query
# print("Similar texts found:")
# for result in results:
#     print(result.page_content)

vector_store.save_local("vector_store")  # Save the vector store to a local directory

vector_store_loaded = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)  # Load the vector store from the local directory
print("Vector store loaded from local directory. Performing similarity search again:")
results_loaded = vector_store_loaded.similarity_search(query, k=2)  # Retrieve the top 2 most similar texts to the query from the loaded vector store
for result in results_loaded:
    print(result.page_content)