
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

QUERY_EXPANSIONS = {
    "library": "books borrowing lending",
    "hours": "opening closing open timing schedule",
    "time": "timing hours schedule",
    "scholarship": "financial aid merit award",
    "gpa": "grade point average academic score",
    "deadline": "due date last date submit submission",
    "register": "registration enrollment enroll",
    "registration": "register enrollment enroll",
    "exam": "examination test assessment",
    "exams": "examination test assessment",
    "hall ticket": "admit card examination entry",
    "identity card": "college id student identification",
    "student welfare": "financial aid scholarship",
    "office": "location room building address",
    "fine": "penalty late fee charge",
    "borrow": "borrowing issue books loan",
    "duration": "period days return due",
}

TIME_RE = re.compile(
    r"\b(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)"
    r"|\d{1,2}:\d{2})\b",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    page: int | None = None


class CollegeRAG:
    def __init__(
        self,
        documents_dir: str | Path,
        min_score: float = 0.10,
        chunk_size: int = 900,
        overlap: int = 150,
    ):
        self.documents_dir = Path(documents_dir)
        self.min_score = min_score
        self.chunk_size = max(250, chunk_size)
        self.overlap = min(max(0, overlap), self.chunk_size // 3)

        self.chunks: list[Chunk] = []
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.word_matrix = None
        self.char_matrix = None
        self._indexed = False

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[\u2010-\u2015]", "-", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _read_file(path: Path) -> list[tuple[int | None, str]]:
        if path.suffix.lower() == ".txt":
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            return [(None, text)]

        if path.suffix.lower() == ".pdf":
            pages = []
            with pymupdf.open(path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    pages.append((page_number, page.get_text("text")))
            return pages

        return []

    def _split_text(self, text: str) -> list[str]:
        text = text.replace("\x00", " ").strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                boundary = max(
                    text.rfind("\n\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind("\n", start, end),
                )
                if boundary > start + int(self.chunk_size * 0.55):
                    end = boundary + 1

            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)

            if end >= len(text):
                break

            start = max(start + 1, end - self.overlap)

        return chunks

    def ingest(self) -> int:
        if not self.documents_dir.exists():
            raise FileNotFoundError(
                f"Documents folder not found: {self.documents_dir}"
            )

        self.chunks = []

        paths = sorted(
            path
            for path in self.documents_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        for path in paths:
            try:
                page_texts = self._read_file(path)
            except Exception as exc:
                print(f"Skipping {path.name}: {exc}")
                continue

            source = str(path.relative_to(self.documents_dir))

            for page_number, page_text in page_texts:
                for part in self._split_text(page_text):
                    self.chunks.append(
                        Chunk(
                            text=part,
                            source=source,
                            chunk_id=len(self.chunks) + 1,
                            page=page_number,
                        )
                    )

        if not self.chunks:
            self._indexed = False
            raise ValueError(
                f"No readable PDF or TXT documents found in "
                f"{self.documents_dir}"
            )

        corpus = [self._normalise(chunk.text) for chunk in self.chunks]

        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9.-]*\b",
        )
        self.word_matrix = self.word_vectorizer.fit_transform(corpus)

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.char_matrix = self.char_vectorizer.fit_transform(corpus)

        self._indexed = True
        return len(self.chunks)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _expanded_query(self, question: str) -> str:
        normalized = self._normalise(question)
        expansions = []

        for term, expansion in QUERY_EXPANSIONS.items():
            if term in normalized:
                expansions.append(expansion)

        return normalized + " " + " ".join(expansions)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        if not self._indexed:
            self.ingest()

        if not question or not question.strip():
            return []

        query = self._expanded_query(question)
        threshold = self.min_score if min_score is None else min_score

        word_query = self.word_vectorizer.transform([query])
        char_query = self.char_vectorizer.transform([query])

        word_scores = cosine_similarity(
            word_query, self.word_matrix
        ).ravel()
        char_scores = cosine_similarity(
            char_query, self.char_matrix
        ).ravel()

        question_tokens = self._tokens(question)
        expanded_tokens = self._tokens(query)

        ranked = []

        for index, chunk in enumerate(self.chunks):
            chunk_tokens = self._tokens(chunk.text)
            if not chunk_tokens:
                continue

            exact_overlap = (
                len(question_tokens & chunk_tokens)
                / max(1, len(question_tokens))
            )
            expanded_overlap = (
                len(expanded_tokens & chunk_tokens)
                / max(1, len(expanded_tokens))
            )

            score = (
                0.60 * float(word_scores[index])
                + 0.20 * float(char_scores[index])
                + 0.15 * exact_overlap
                + 0.05 * expanded_overlap
            )

            ranked.append({
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "score": round(score, 4),
                "word_score": round(float(word_scores[index]), 4),
                "char_score": round(float(char_scores[index]), 4),
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)

        results = []
        seen_source_text = set()
        source_counts: dict[str, int] = {}

        for item in ranked:
            if item["score"] < threshold:
                continue

            dedup_key = (
                item["source"],
                self._normalise(item["text"]),
            )
            if dedup_key in seen_source_text:
                continue
            seen_source_text.add(dedup_key)

            if source_counts.get(item["source"], 0) >= 2:
                continue

            results.append(item)
            source_counts[item["source"]] = (
                source_counts.get(item["source"], 0) + 1
            )

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _library_hours_conflict(
        question: str,
        chunks: list[Chunk],
    ) -> list[dict[str, Any]]:
        q = question.lower()

        asks_library_hours = (
            "library" in q
            and any(
                word in q
                for word in ("hour", "open", "close", "timing")
            )
        )
        if not asks_library_hours:
            return []

        by_source: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if "library" in chunk.text.lower():
                by_source.setdefault(chunk.source, []).append(chunk)

        evidence = []

        for source, source_chunks in by_source.items():
            combined = " ".join(chunk.text for chunk in source_chunks)
            lower = combined.lower()

            if not any(
                word in lower
                for word in ("hour", "open", "opening", "closed")
            ):
                continue

            times = {
                re.sub(r"\s+", "", match.lower().replace(".", ""))
                for match in TIME_RE.findall(combined)
            }

            if len(times) >= 2:
                evidence.append({
                    "source": source,
                    "text": combined[:1800],
                    "times": sorted(times),
                    "chunk_id": source_chunks[0].chunk_id,
                    "page": source_chunks[0].page,
                })

        if len(evidence) < 2:
            return []

        distinct_sets = {tuple(item["times"]) for item in evidence}
        return evidence if len(distinct_sets) > 1 else []

    @staticmethod
    def _format_citation(source: dict[str, Any]) -> str:
        location = f"chunk {source['chunk_id']}"
        if source.get("page") is not None:
            location += f", page {source['page']}"
        return f"{source['source']} ({location})"

    @staticmethod
    def _insufficient_evidence(message: str) -> dict[str, Any]:
        return {
            "status": "insufficient_evidence",
            "answer": message,
            "sources": [],
            "conflicts": [],
            "confidence": 0.0,
        }

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        # Blank-question behavior expected by the tests.
        if not question or not question.strip():
            return self._insufficient_evidence(
                "No question was provided. Please enter a question "
                "about the college documents."
            )

        if not self._indexed:
            self.ingest()

        conflicts = self._library_hours_conflict(question, self.chunks)

        if conflicts:
            sources = [
                {
                    "source": item["source"],
                    "text": item["text"],
                    "chunk_id": item["chunk_id"],
                    "page": item["page"],
                    "score": 1.0,
                }
                for item in conflicts
            ]

            details = "\n".join(
                f"- **{item['source']}** reports "
                f"{', '.join(item['times'])}."
                for item in conflicts
            )

            return {
                "status": "conflicting_evidence",
                "answer": (
                    "The documents contain conflicting library-hours "
                    "information, so I cannot determine which version is "
                    "currently authoritative.\n\n"
                    f"{details}\n\n"
                    "Please confirm the current hours with the college library."
                ),
                "sources": sources,
                "conflicts": conflicts,
                "confidence": 1.0,
            }

        passages = self.retrieve(question, top_k=top_k)

        # Require both a reasonable score and actual lexical overlap.
        # This reduces answers to unrelated questions that happen to score
        # well because of query expansion or character similarity.
        answer_threshold = max(self.min_score, 0.15)
        question_tokens = self._tokens(question)

        best_passage = passages[0] if passages else None
        best_tokens = (
            self._tokens(best_passage["text"])
            if best_passage else set()
        )
        meaningful_overlap = question_tokens & best_tokens

        # Common question words alone are not meaningful evidence.
        generic_tokens = {
            "what", "where", "when", "who", "why", "how",
            "is", "are", "the", "a", "an", "of", "in", "on",
            "for", "to", "and", "or", "do", "does", "can",
            "i", "my", "me", "please", "tell",
        }
        meaningful_overlap -= generic_tokens

        if (
            best_passage is None
            or best_passage["score"] < answer_threshold
            or not meaningful_overlap
        ):
            return self._insufficient_evidence(
                "I couldn't find sufficiently relevant evidence in the "
                "uploaded college documents. Please rephrase your question "
                "or check with the relevant college office."
            )

        answer_lines = [
            "Here is the most relevant information found in your documents:"
        ]

        for index, passage in enumerate(passages[:3], start=1):
            text = re.sub(r"\s+", " ", passage["text"]).strip()
            answer_lines.append(
                f"\n{index}. {text}\n"
                f"   Source: {self._format_citation(passage)}"
            )

        return {
            "status": "answered",
            "answer": "\n".join(answer_lines),
            "sources": passages,
            "conflicts": [],
            "confidence": round(best_passage["score"], 3),
        }
