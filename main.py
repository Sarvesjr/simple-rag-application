from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(persist_directory="chroma-db", embedding_function=embedding_model)

retriever = vectorstore.as_retriever(search_type = "mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5})

llm = ChatMistralAI(model="ministral-3b-2410")

#prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use only the information provided context to answer the question. If the answer is not in the context, say you could not find the answer."),
    ("human", "{context}\n\nQuestion: {question}\nAnswer:")
])

print("RAG system is ready. You can ask questions now.")

print("Type 0 to exit.")

while True:
    query = input("\nEnter your question: ")
    if query == "0":
        print("Exiting the RAG system.")
        break

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query,
    })

    response = llm.invoke(final_prompt)
    print("\nAnswer:", response.content)