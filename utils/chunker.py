"""
Document chunking utilities for text processing (TXT-only) using local embeddings.
"""
import os
from typing import Dict, Any, List, Optional, Iterable
from pathlib import Path
import spacy
import numpy as np
import json

from .embedding_local import get_embedding
from .logger import get_logger

logger = get_logger(__name__)

# Load spaCy lazily (avoid heavy import-time cost in some contexts).
_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
    return _NLP


# Defaults
_DEFAULT_CHUNK_SIZE = int(os.getenv("INGESTION_CHUNK_SIZE", 1000))  # tokens (heuristic)
_DEFAULT_CHUNK_OVERLAP_SENTENCES = int(os.getenv("INGESTION_CHUNK_OVERLAP_SENTENCES", 3))
_DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("INGESTION_SIMILARITY_THRESHOLD", 0.6))
_DEFAULT_EMBED_BATCH = int(os.getenv("INGESTION_EMBED_BATCH", 32))


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate — currently words. Replace with a true tokenizer (e.g. tiktoken) if you need precision.
    """
    return len(text.split()) if text else 0

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate the text so it doesn't exceed max_tokens.
    Simple heuristic: split by spaces (words) and cut at max_tokens.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def get_embeddings_batch(sentences: Iterable[str], batch_size: int = _DEFAULT_EMBED_BATCH) -> np.ndarray:
    """
    Get embeddings for a list/iterable of sentences. Batches calls to get_embedding to avoid making a call per sentence.
    Assumes get_embedding returns a 1D numpy array or list-like.
    Returns a (N, D) numpy array of L2-normalized vectors (cosine-ready).
    """
    embs: List[np.ndarray] = []
    batch: List[str] = []
    for s in sentences:
        batch.append(s)
        if len(batch) >= batch_size:
            # call get_embedding for each in batch (if your embedding provider has real batching, replace here)
            embs.extend([np.array(get_embedding(x), dtype=np.float32) for x in batch])
            batch = []
    if batch:
        embs.extend([np.array(get_embedding(x), dtype=np.float32) for x in batch])

    if not embs:
        return np.zeros((0, 0), dtype=np.float32)

    embs_array = np.vstack(embs).astype(np.float32)
    embs_array = _normalize_rows(embs_array)
    return embs_array


def semantic_chunk_sentences(
    sentences: List[str],
    max_chunk_tokens: int = _DEFAULT_CHUNK_SIZE,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
) -> List[str]:
    """
    Build semantically-coherent chunks by iterating sentence-by-sentence and either appending to the current chunk
    or starting a new one when token budget is exceeded or similarity to centroid drops below threshold.

    NOTE: similarity uses cosine similarity; both sentence embeddings and centroid are normalized.
    """
    if not sentences:
        return []

    sent_embs = get_embeddings_batch(sentences)
    if sent_embs.size == 0:
        # fallback: chunk by naive token size if embeddings failed
        logger.warning("No embeddings returned; falling back to naive token-based sentence join.")
        chunks: List[str] = []
        cur: List[str] = []
        cur_toks = 0
        for s in sentences:
            s_toks = estimate_tokens(s)
            if cur and cur_toks + s_toks > max_chunk_tokens:
                chunks.append(" ".join(cur))
                cur = [s]
                cur_toks = s_toks
            else:
                cur.append(s)
                cur_toks += s_toks
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    chunks: List[str] = []
    current_chunk_sents: List[str] = []
    current_embs: List[np.ndarray] = []

    for i, sent in enumerate(sentences):
        emb = sent_embs[i]

        if not current_chunk_sents:
            current_chunk_sents.append(sent)
            current_embs.append(emb)
            continue

        # centroid: mean then normalize
        centroid = np.mean(np.vstack(current_embs), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        sim = float(np.dot(emb, centroid))  # cosine similarity

        tentative_text = " ".join(current_chunk_sents + [sent])
        tentative_tokens = estimate_tokens(tentative_text)

        # Conditions to break: token overflow OR (low similarity AND chunk not too small)
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
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP_SENTENCES,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
) -> List[str]:
    """
    Chunk text. Two modes:
    - If the text looks like lots of short lines (e.g., < 50 chars), split by line length into character-based chunks.
    - Otherwise, split into semantic chunks by sentences.

    chunk_overlap indicates how many sentences from previous chunk to prefix onto the next chunk (sentence-level overlap).
    """
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # short-line heuristic
    short_line_ratio = sum(1 for line in lines if len(line) < 50) / len(lines)
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
    nlp = get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = semantic_chunk_sentences(
        sentences,
        max_chunk_tokens=chunk_size,
        similarity_threshold=similarity_threshold
    )

    # filter tiny chunks
    chunks = [c for c in chunks if len(c.strip()) > 10]

    if chunk_overlap and chunk_overlap > 0:
        final_chunks: List[str] = []
        parsed_sent_lists = [list(get_nlp()(c).sents) for c in chunks]
        for i, chunk in enumerate(chunks):
            if i == 0:
                final_chunks.append(chunk)
                continue
            prev_sents = parsed_sent_lists[i - 1]
            overlap_n = min(len(prev_sents), chunk_overlap)
            overlap_text = " ".join([s.text for s in prev_sents[-overlap_n:]]).strip()
            combined = f"{overlap_text} {chunk}".strip() if overlap_text else chunk
            final_chunks.append(combined)
        return final_chunks

    return chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP_SENTENCES,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    documents: list of dicts with either:
      - {"file_path": "<path>", "metadata": {...}}  OR
      - {"content": "<text>", "metadata": {...}}
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        if "file_path" in doc:
            file_path = Path(doc["file_path"])
            if not file_path.exists():
                logger.error("File not found: %s", file_path)
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            metadata = {**doc.get("metadata", {}), "source": str(file_path), "file_type": file_path.suffix}
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


if __name__ == "__main__":
    out_folder = Path(__file__).parent / "out"
    if not out_folder.exists():
        raise FileNotFoundError(f"Could not find 'out' folder: {out_folder}")

    txt_files = list(out_folder.glob("*.txt"))
    if not txt_files:
        logger.info("No TXT files found in the 'out' folder.")
        raise SystemExit(0)

    for txt_file in txt_files:
        logger.info("Loading and chunking: %s", txt_file.name)
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        docs = [{"content": text, "metadata": {"source": str(txt_file), "file_type": ".txt"}}]
        chunks = chunk_documents(docs)

        logger.info("Created %d chunks from %s", len(chunks), txt_file.name)

        output_file = out_folder / f"{txt_file.stem}_chunks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info("Chunks saved to: %s", output_file)
