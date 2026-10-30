
# College Knowledge Assistant

## Overview

A Python-based RAG-style assistant that retrieves information from
college documents and returns extractive answers with source citations.

## Technologies

- Python
- Streamlit
- scikit-learn
- NumPy
- PyMuPDF
- pytest

No paid LLM API is required.

## Setup on Windows

Activate your existing virtual environment:

    DLenv\Scripts\activate

Install dependencies:

    python -m pip install -r requirements.txt

Place at least five PDF or TXT files in the documents folder.

Run the application:

    python -m streamlit run app.py

Run automated tests:

    python -m pytest tests\test_rag.py -v

## Architecture

1. Read PDF and TXT documents.
2. Extract text and preserve source metadata.
3. Split documents into searchable chunks.
4. Build a TF-IDF vector index.
5. Rank passages using cosine similarity.
6. Check for likely date/time conflicts.
7. Return extracted passages and source citations.
8. Abstain if no relevant passage is retrieved.

## Testing

Automated tests use the separate test_documents directory.

The tests cover direct answers, questions requiring multiple sources,
unanswerable questions, paraphrased questions, conflicting documents,
source metadata, library policies, scholarship documents, and examinations.

## Limitations

TF-IDF matches lexical patterns and is not a deep semantic model.
Query expansion helps common paraphrases but does not guarantee retrieval
of every equivalent question.

Conflict detection is heuristic. It does not establish document authority
or resolve every type of contradiction.

Scanned PDFs may require OCR.

## Demo

1. Start the application.
2. Ask a direct question.
3. Inspect the supporting source.
4. Ask a question requiring two documents.
5. Ask an unrelated question.
6. Ask about the conflicting library hours.
7. Upload an unseen document set.
8. Explain retrieval ranking, the relevance threshold, and a limitation.

The included sample policies are fictional test data.
Replace them with approved official college documents for real use.
