from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter 

splitter = CharacterTextSplitter(chunk_size=10,
                                 separator="\n\n",
                                chunk_overlap=1)

data = TextLoader("document loaders/sample.txt")

docs = data.load()

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()