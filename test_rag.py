
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_pipeline import CollegeRAG

# Use the folder containing your six uploaded sample documents.
DOCS = ROOT / "test_documents"


@pytest.fixture(scope="module")
def rag():
    instance = CollegeRAG(
        documents_dir=DOCS,
        min_score=0.08,
    )
    instance.ingest()
    return instance


# Test 1: Answer appears clearly in one document.
def test_1_direct_answer(rag):
    result = rag.answer("When does the semester begin?")

    assert result["status"] == "answered"
    assert "5 August 2026" in result["answer"]
    assert any(
        item["source"] == "Database-I.txt"
        for item in result["sources"]
    )


# Test 2: Answer requires two different documents.
def test_2_information_from_two_documents(rag):
    result = rag.answer(
        "What GPA is required for a merit scholarship, "
        "and where is the student welfare office?"
    )

    assert result["status"] == "answered"

    sources = {item["source"] for item in result["sources"]}

    assert "Database-IV.txt" in sources
    assert "Database-V.txt" in sources


# Test 3: Unrelated question must be rejected.
def test_3_unrelated_question_abstains(rag):
    result = rag.answer("Who won the 2022 FIFA World Cup?")

    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


# Test 4: Different wording for a known fact.
def test_4_paraphrased_question(rag):
    result = rag.answer(
        "By what date must I enroll in semester courses?"
    )

    assert result["status"] == "answered"
    assert "12 August 2026" in result["answer"]


# Test 5: Conflicting library hours must be detected.
def test_5_conflicting_library_hours(rag):
    result = rag.answer("What time does the library close?")

    assert result["status"] == "conflicting_evidence"

    sources = {item["source"] for item in result["sources"]}

    assert "Database-II.txt" in sources
    assert "Database-VI.txt" in sources


# Test 6: Library late-return fine.
def test_library_fine(rag):
    result = rag.answer("What is the fine for a late library book?")

    assert result["status"] == "answered"
    assert any(
        "5 rupees" in item["text"]
        for item in result["sources"]
    )


# Test 7: Library borrowing limit.
def test_library_borrowing_limit(rag):
    result = rag.answer("How many books may I borrow?")

    assert result["status"] == "answered"
    assert any(
        "three books" in item["text"]
        for item in result["sources"]
    )


# Test 8: Library loan duration.
def test_library_loan_duration(rag):
    result = rag.answer("How long can I keep borrowed books?")

    assert result["status"] == "answered"
    assert any(
        "fourteen days" in item["text"]
        for item in result["sources"]
    )


# Test 9: Examination requirements.
def test_examination_requirements(rag):
    result = rag.answer("What must I bring to the exam?")

    assert result["status"] == "answered"

    evidence = " ".join(
        item["text"] for item in result["sources"]
    ).lower()

    assert "identity card" in evidence
    assert "hall ticket" in evidence


# Test 10: Examination arrival time.
def test_examination_arrival_time(rag):
    result = rag.answer(
        "How early should I arrive at the examination hall?"
    )

    assert result["status"] == "answered"
    assert any(
        "30 minutes" in item["text"]
        for item in result["sources"]
    )


# Test 11: Scholarship supporting documents.
def test_scholarship_documents(rag):
    result = rag.answer(
        "Which documents are required for the merit scholarship?"
    )

    assert result["status"] == "answered"

    evidence = " ".join(
        item["text"] for item in result["sources"]
    ).lower()

    assert "grade report" in evidence
    assert "family income" in evidence


# Test 12: Another unanswerable question.
def test_unrelated_science_question(rag):
    result = rag.answer("Explain quantum entanglement.")

    assert result["status"] == "insufficient_evidence"


# Test 13: Citation metadata is preserved.
def test_source_metadata(rag):
    result = rag.answer("When does the semester begin?")

    assert result["sources"]

    for source in result["sources"]:
        assert source["source"]
        assert "page" in source
        assert "text" in source
        assert "score" in source


# Test 14: Empty question.
def test_blank_question(rag):
    result = rag.answer("   ")

    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []
