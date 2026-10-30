
from pathlib import Path
import hashlib

import streamlit as st

from rag_pipeline import CollegeRAG


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_documents():
    """Return all supported documents in the knowledge-base folder."""
    return sorted(
        [
            path
            for path in DOCUMENTS_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def collection_signature():
    """
    Create a signature from filenames, sizes, and modification times.
    Changes when documents are added, removed, or modified.
    """
    records = []

    for path in get_documents():
        stat = path.stat()
        records.append(
            f"{path.relative_to(DOCUMENTS_DIR)}:"
            f"{stat.st_size}:{stat.st_mtime_ns}"
        )

    raw = "\n".join(records)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_uploaded_files(uploaded_files):
    """Save uploaded files to the knowledge-base directory."""
    saved_count = 0
    skipped_count = 0

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            skipped_count += 1
            continue

        # Avoid directory traversal and nested paths from uploaded filenames.
        safe_name = Path(uploaded_file.name).name
        destination = DOCUMENTS_DIR / safe_name

        destination.write_bytes(uploaded_file.getvalue())
        saved_count += 1

    return saved_count, skipped_count


@st.cache_resource(show_spinner=False)
def build_knowledge_base(signature: str):
    """
    Build an index for the current collection.
    The signature changes whenever the saved collection changes.
    """
    rag = CollegeRAG(
        documents_dir=DOCUMENTS_DIR,
        min_score=0.10,
    )
    chunk_count = rag.ingest()

    return rag, chunk_count


def load_knowledge_base():
    """Load the index for the current collection."""
    signature = collection_signature()
    return build_knowledge_base(signature)


def display_sources(sources):
    """Display supporting passages without duplicate source expanders."""
    if not sources:
        st.info("No supporting passages were retrieved.")
        return

    st.subheader("Supporting sources")

    grouped_sources = {}

    for source in sources:
        filename = source.get("source", "Unknown source")
        grouped_sources.setdefault(filename, []).append(source)

    for filename, passages in grouped_sources.items():
        # Sort the most relevant passage first.
        passages.sort(
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )

        with st.expander(f"📄 {filename}", expanded=False):
            for index, passage in enumerate(passages, start=1):
                chunk_id = passage.get("chunk_id")
                page = passage.get("page")
                score = passage.get("score")

                details = []
                if chunk_id is not None:
                    details.append(f"Chunk {chunk_id}")
                if page is not None:
                    details.append(f"Page {page}")
                if score is not None:
                    details.append(f"Relevance score: {score:.3f}")

                if details:
                    st.caption(" · ".join(details))

                st.markdown(passage.get("text", ""))

                if index < len(passages):
                    st.divider()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎓 College Knowledge Assistant")

st.write(
    "Ask questions about college rules, academic dates, examinations, "
    "scholarships, library policies, and student services."
)

st.info(
    "Answers are extracted from retrieved documents. "
    "Review the supporting passages and verify official policies."
)


# --------------------------------------------------
# Sidebar: knowledge base management
# --------------------------------------------------

with st.sidebar:
    st.header("Knowledge base")

    st.write("Upload PDF or TXT files to create your college document collection.")

    uploaded_files = st.file_uploader(
        "College documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="college_document_uploader",
    )

    if st.button("Save uploaded documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Select at least one PDF or TXT file first.")
        else:
            try:
                saved, skipped = save_uploaded_files(uploaded_files)

                # The changed collection signature triggers a fresh index.
                st.session_state.pop("last_answer", None)

                if saved:
                    st.success(f"Saved {saved} document(s).")
                if skipped:
                    st.warning(f"Skipped {skipped} unsupported file(s).")

                st.rerun()

            except OSError as exc:
                st.error(f"Could not save uploaded files: {exc}")

    st.divider()
    st.subheader("Saved documents")

    saved_documents = get_documents()

    if saved_documents:
        for path in saved_documents:
            relative_name = str(path.relative_to(DOCUMENTS_DIR))
            file_size = path.stat().st_size

            st.caption(f"📄 {relative_name} · {file_size:,} bytes")

            if st.button(
                f"Remove {path.name}",
                key=f"remove_{relative_name}",
                use_container_width=True,
            ):
                try:
                    path.unlink()
                    st.session_state.pop("last_answer", None)
                    st.success(f"Removed {path.name}.")
                    st.rerun()
                except OSError as exc:
                    st.error(f"Could not remove {path.name}: {exc}")
    else:
        st.caption("No documents uploaded yet.")

    st.divider()

    if st.button("Build / refresh knowledge base", use_container_width=True):
        try:
            # Explicitly clear cached indexes, then rebuild.
            build_knowledge_base.clear()
            with st.spinner("Building knowledge base..."):
                rag, count = load_knowledge_base()

            st.session_state["indexed_signature"] = collection_signature()
            st.session_state.pop("last_answer", None)

            st.success(f"Knowledge base refreshed: {count} chunks indexed.")
        except Exception as exc:
            st.error(f"Could not build knowledge base: {exc}")

    st.caption(
        "Supported files: PDF and TXT. "
        "Uploading the same filename replaces the existing file."
    )


# --------------------------------------------------
# Knowledge-base status
# --------------------------------------------------

current_documents = get_documents()

if not current_documents:
    st.warning(
        "Your knowledge base is empty. Upload at least one PDF or TXT file "
        "using the sidebar, then click Build / refresh knowledge base."
    )
    st.stop()

try:
    current_signature = collection_signature()

    with st.spinner("Loading college knowledge base..."):
        rag, chunk_count = load_knowledge_base()

    st.caption(
        f"Indexed chunks: {chunk_count} | "
        f"Documents: {len(current_documents)} | "
        f"Collection: {DOCUMENTS_DIR.name}"
    )

except Exception as exc:
    st.error(f"Could not load the knowledge base: {exc}")
    st.stop()


# --------------------------------------------------
# Question and answer
# --------------------------------------------------

st.subheader("Ask a question")

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the library hours?",
    )

    submitted = st.form_submit_button(
        "Find answer",
        type="primary",
        use_container_width=False,
    )

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching college documents..."):
                result = rag.answer(question.strip(), top_k=5)

            st.session_state["last_answer"] = {
                "question": question.strip(),
                "result": result,
                "signature": current_signature,
            }

        except Exception as exc:
            st.error(f"An error occurred while answering your question: {exc}")


# --------------------------------------------------
# Render the last answer
# --------------------------------------------------

saved_answer = st.session_state.get("last_answer")

if saved_answer:
    if saved_answer.get("signature") != current_signature:
        st.info("The document collection changed. Submit your question again.")
    else:
        result = saved_answer["result"]
        status = result.get("status", "answered")
        answer = result.get("answer", "No answer was returned.")

        st.divider()

        if status == "conflicting_evidence":
            st.warning(answer)
        elif status in {"insufficient_evidence", "invalid_question"}:
            st.info(answer)
        else:
            st.subheader("Answer")
            st.markdown(answer)

        display_sources(result.get("sources", []))

        if status == "answered":
            confidence = result.get("confidence")
            if confidence is not None:
                st.caption(
                    f"Retrieval score indicator: {confidence:.3f}. "
                    "This is a ranking score, not a probability of correctness."
                )
