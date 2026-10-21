
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"}
    ),
]

# Initialize Mistral AI embeddings
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# Create the Chroma vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

# Similarity search
result = vectorstore.similarity_search(
    "what is used for data analysis?",
    k=2
)

for r in result:
    print(r.page_content)
    print(r.metadata)

# Create a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Retrieve documents related to deep learning
retrieved_docs = retriever.invoke("Explain deep learning")

for d in retrieved_docs:
    print(d.page_content)
    print(d.metadata)
