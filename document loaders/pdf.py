from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

data = PyPDFLoader("document loaders/deeplearning.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk.page_content)
    print("Source:", chunk.metadata.get("source"))
    print("Page:", chunk.metadata.get("page"))