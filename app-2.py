
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

st.set_page_config(
    page_title="C++ RAG Assistant",
    page_icon="📘",
    layout="wide",
)

st.title("📘 C++ RAG Assistant")
st.write("Ask questions about your C++ notes using Gemma 2 and Ollama.")

# Document path
DOC_PATH = Path(__file__).parent / "Cpp.txt"

# Sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value="gemma2:2b")
    top_k = st.slider("Retrieved passages", 1, 6, 3)

    uploaded_file = st.file_uploader(
        "Upload a text document",
        type=["txt"],
    )


# Read document
if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode(
        "utf-8", errors="replace"
    )
    source_name = uploaded_file.name
elif DOC_PATH.exists():
    document_text = DOC_PATH.read_text(
        encoding="utf-8", errors="replace"
    )
    source_name = DOC_PATH.name
else:
    st.error("Cpp.txt not found. Upload your text file in the sidebar.")
    st.stop()


@st.cache_resource(show_spinner="Building document search index...")
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    documents = splitter.split_documents([
        Document(page_content=text)
    ])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents, embeddings)


try:
    vectorstore = create_vectorstore(document_text)
except Exception as exc:
    st.error(f"Could not create the search index: {exc}")
    st.stop()


# Question input
question = st.text_input(
    "Enter your C++ question",
    placeholder="What is polymorphism in C++?",
)

if st.button("Search and Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner("Searching your document..."):
            results = vectorstore.similarity_search(
                question,
                k=top_k,
            )

        if not results:
            st.warning("No relevant passages were found.")
            st.stop()

        context = "\n\n".join(
            f"Passage {i}:\n{doc.page_content}"
            for i, doc in enumerate(results, start=1)
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful C++ question-answering assistant.
Answer using the supplied document context.
If the context does not contain the answer, say so.
Do not invent information.
Explain the answer clearly and include a short example
when the context supports one."""
            ),
            (
                "human",
                "Document context:\n{context}\n\nQuestion:\n{question}"
            ),
        ])

        llm = ChatOllama(
            model=model_name.strip(),
            temperature=0,
        )

        chain = prompt | llm

        with st.spinner(f"Gemma ({model_name}) is generating an answer..."):
            response = chain.invoke({
                "context": context,
                "question": question,
            })

        st.subheader("Answer")
        st.markdown(response.content)

        with st.expander("View retrieved document passages"):
            for i, doc in enumerate(results, start=1):
                st.markdown(f"**Passage {i}**")
                st.write(doc.page_content)

    except Exception as exc:
        st.error("Could not complete the question.")
        st.code(str(exc))
        st.info(
            "Check that Ollama is running and that the selected "
            "model is installed. Run `ollama list` to verify."
        )
