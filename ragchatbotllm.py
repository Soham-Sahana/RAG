import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="C++ RAG AI assiant", layout="wide")
st.title("C++ RAG AI Assistant")

@st.cache_resource
def load_vectorstore():
    loader = TextLoader("Cpp.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
    
    db = FAISS.from_documents(final_documents, embeddings)
    return db

db = load_vectorstore()

llm = OllamaLLM(model="gemma2:2b")

query = st.text_input("Enter your question : ")
if query:
    with st.spinner("Thinking..."):
        docx = db.similarity_search(query)
        context = "\n".join(docs.page_content for docs in docx )

        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        response = llm.invoke(prompt)
        st.subheader("Answer:")
        st.write(response)



        