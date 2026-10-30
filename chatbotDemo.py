import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="C++ RAG AI assiant")
st.title("C++ RAG CHATBOT")
st.write("Ask any questions related to C++ introduction.")

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

query = st.text_input("Enter your question : ")
if query:
    docx = db.similarity_search(query,k=3)
    st.subheader("Retrieved Context:")
    for i, doc in enumerate(docx):
       st.markdown(f"**Result {i+1}:**")
       st.write(doc.page_content)