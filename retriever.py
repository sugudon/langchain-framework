from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


text = [
    "I love my buji and scheru", 
    "They are like my children", 
    "I love programming in Python", 
    "LangChain is a great framework for building applications with LLMs", 
    "I enjoy hiking and outdoor activities", 
    "My favorite programming language is Python", 
    "I have two adorable pets, a buji and a scheru", 
    "LangChain simplifies the process of building applications with LLMs"]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(text, embeddings)

retrieve_query = "Tell me about your pets"

retrieved_results = vector_store.as_retriever(search_kwargs={"k": 2})# Retrieve the top 2 most similar texts to the query

results = retrieved_results.invoke(retrieve_query)  # Use the retriever to perform the similarity search

for result in results:
    print(result.page_content)