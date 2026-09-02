from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("example.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages.")
print(f"First page metadata: {pages[0].metadata}")
print(f"First page content: {pages[0].page_content}")