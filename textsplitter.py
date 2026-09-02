from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


long_text = TextLoader("long_text.txt").load()

long_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
long_text_chunks = long_text_splitter.split_text(long_text[0].page_content)

print(f"Loaded {len(long_text_chunks)} chunks.")

for i, chunk in enumerate(long_text_chunks):
    print(f"\n Chunk {i + 1} content: {chunk}")


long_text_from_documents = long_text_splitter.split_documents(long_text)

print(f"\nLoaded {len(long_text_from_documents)} chunks from documents.")
for i, chunk in enumerate(long_text_from_documents):
    print(f"\n Chunk {i + 1} content: {chunk.page_content}")


