"""
Document chunking utilities for text processing (TXT-only).
"""
import os
from typing import Dict, Any, List
from pathlib import Path
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load spaCy model and embedding model (may download the latter once)
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

# Defaults
_DEFAULT_CHUNK_SIZE = int(os.getenv("INGESTION_CHUNK_SIZE", 1000))
_DEFAULT_CHUNK_OVERLAP = int(os.getenv("INGESTION_CHUNK_OVERLAP", 100))

def semantic_chunk_sentences(
    sentences: List[str],
    max_chunk_tokens: int = int(os.getenv("INGESTION_CHUNK_SIZE", 1000)),
    similarity_threshold: float = 0.52,
    embed_batch_size: int = 64
) -> List[str]:
    if not sentences:
        return []

    sent_embs = embed_model.encode(
        sentences,
        batch_size=embed_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    chunks: List[str] = []
    current_chunk_sents: List[str] = []
    current_embs: List[np.ndarray] = []

    for i, sent in enumerate(sentences):
        emb = sent_embs[i]

        if not current_chunk_sents:
            current_chunk_sents.append(sent)
            current_embs.append(emb)
            continue

        centroid = np.mean(np.stack(current_embs, axis=0), axis=0)
        sim = float(np.dot(emb, centroid))  # cosine since normalized

        tentative_text = " ".join(current_chunk_sents + [sent])
        tentative_tokens = estimate_tokens(tentative_text)

        if tentative_tokens > max_chunk_tokens or (sim < similarity_threshold and tentative_tokens > 50):
            chunks.append(" ".join(current_chunk_sents))
            current_chunk_sents = [sent]
            current_embs = [emb]
        else:
            current_chunk_sents.append(sent)
            current_embs.append(emb)

    if current_chunk_sents:
        chunks.append(" ".join(current_chunk_sents))

    return chunks

def chunk_text(
    text: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    similarity_threshold: float = 0.6
) -> List[str]:
    if not text:
        return []

    # quick line-based fallback for files with many short lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_line_ratio = sum(1 for line in lines if len(line) < 50) / max(len(lines), 1)
    if short_line_ratio > 0.6:
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0
        for line in lines:
            line_length = len(line)
            if current_length + line_length + 1 > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length + 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    # sentence-based semantic chunking
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = semantic_chunk_sentences(
        sentences,
        max_chunk_tokens=chunk_size,
        similarity_threshold=similarity_threshold
    )

    # remove trivially short chunks
    chunks = [c for c in chunks if len(c.strip()) > 10]

    # optional sentence-overlap (by naive split on '. ')
    if chunk_overlap > 0:
        final_chunks: List[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                final_chunks.append(chunk)
                continue
            prev_sents = chunks[i - 1].split('. ')
            overlap_n = min(len(prev_sents), chunk_overlap)
            overlap_text = '. '.join(prev_sents[-overlap_n:]).strip()
            if overlap_text and not overlap_text.endswith('.'):
                overlap_text += '.'
            combined = f"{overlap_text} {chunk}".strip() if overlap_text else chunk
            final_chunks.append(combined)
        return final_chunks

    return chunks

def chunk_documents(
    documents: List[Dict[str, Any]],
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    similarity_threshold: float = float(os.getenv("INGESTION_SIMILARITY_THRESHOLD", 0.72))
) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        if "file_path" in doc:
            file_path = Path(doc["file_path"])
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            metadata = {**doc.get("metadata", {}), "source": str(file_path), "file_type": ".txt"}
        else:
            text = doc.get("content", "")
            metadata = doc.get("metadata", {})

        text_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=similarity_threshold
        )

        for i, chunk in enumerate(text_chunks):
            token_count = estimate_tokens(chunk)
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "estimated_tokens": token_count
                }
            })

    return all_chunks

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(nlp(text))

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    out_folder = Path(__file__).parent / "out"
    chunk_size = int(os.getenv("INGESTION_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.getenv("INGESTION_CHUNK_OVERLAP", _DEFAULT_CHUNK_OVERLAP))
    ignore_files = {"confidential_contacts.txt"}

    if not out_folder.exists():
        raise FileNotFoundError(f"Could not find 'out' folder: {out_folder}")

    txt_files = list(out_folder.glob("*.txt"))
    if not txt_files:
        print("No TXT files found in the 'out' folder.")
        raise SystemExit(0)

    for txt_file in txt_files:
        if txt_file.name in ignore_files:
            print(f"Skipping confidential file: {txt_file.name}")
            continue

        print(f"Loading and chunking: {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        docs = [{"content": text, "metadata": {"source": str(txt_file), "file_type": ".txt"}}]
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        print(f"Created {len(chunks)} chunks from {txt_file.name}")

        output_file = out_folder / f"{txt_file.stem}_chunks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Chunks saved to: {output_file}")
