"""
Document chunking utilities for text processing using local embeddings.

Optimizations:
- True batch embedding 
- Reduced spaCy re-parsing overhead
- Sentence caching to avoid redundant NLP calls
- Section-aware chunking for HTML documents (chunks by headers)
"""
import os
import re
from typing import Dict, Any, List, Optional, Iterable, Tuple
from pathlib import Path
import spacy
import numpy as np
import json

from .embedding_local import get_embeddings_batch as _get_embeddings_batch_from_local
from .logger import get_logger

logger = get_logger(__name__)

# Load spaCy lazily (avoid heavy import-time cost in some contexts).
_NLP = None


def get_nlp():
    """Get cached spaCy model instance."""
    global _NLP
    if _NLP is None:
        model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")
        logger.info(f"Loading spaCy model: {model_name}")
        _NLP = spacy.load(model_name)
    return _NLP


# Defaults - read once at module load
_DEFAULT_CHUNK_SIZE = int(os.getenv("INGESTION_CHUNK_SIZE", 1000))  # tokens (heuristic)
_DEFAULT_CHUNK_OVERLAP_SENTENCES = int(os.getenv("INGESTION_CHUNK_OVERLAP_SENTENCES", 3))
_DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("INGESTION_SIMILARITY_THRESHOLD", 0.6))
_DEFAULT_EMBED_BATCH = int(os.getenv("INGESTION_EMBED_BATCH", 32))

# Section-aware chunking patterns (for IT docs)
_SECTION_HEADER_PATTERN = re.compile(
    r'^(?:\*{2,})?'  # Optional markdown bold
    r'(?:'
    r'#{1,6}\s+|'  # Markdown headers
    r'[A-Z][A-Za-z0-9\s\-:]+(?:\n|$)|'  # Title Case headers
    r'[A-Z][A-Z0-9\s\-:]{3,}(?:\n|$)'  # ALL CAPS headers
    r')'
    r'(?:\*{2,})?',  # Optional markdown bold
    re.MULTILINE
)


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


def _extract_sections(text: str) -> List[Tuple[str, str]]:
    """
    Extract sections from text based on headers.
    
    For IT docs, this splits content by logical sections like:
    - "VPN Setup"
    - "Troubleshooting"
    - "Known Issues"
    
    Returns:
        List of (section_title, section_content) tuples
    """
    lines = text.split('\n')
    sections: List[Tuple[str, str]] = []
    current_title = "Overview"
    current_content: List[str] = []
    
    # Patterns that are NOT headers (table fragments, status codes, etc.)
    non_header_patterns = [
        r'^\d{1,2}[:.]\d{2}',  # Times like "10.00-14.00"
        r'^[A-Z]{2,6}\s+(DONE|TODO|N/A)',  # Status codes like "KESK DONE"
        r'^\+\d+',  # Phone numbers
        r'^https?://',  # URLs
        r'^\*\*[^*]+\*\*$',  # Just bold text (often table cells)
        r'^(Responsible|Completed|Category|Action|Services)',  # Table headers
        r'^(PRA|CU)\s*\d',  # Version numbers
        r'^[\w\s]+(upgrade|update|migrate|move)',  # Action descriptions
    ]
    non_header_re = re.compile('|'.join(non_header_patterns), re.IGNORECASE)
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_content.append(line)
            continue
        
        # Skip if matches non-header patterns
        if non_header_re.search(stripped):
            current_content.append(line)
            continue
        
        # Check if this line is a header
        is_header = False
        header_text = None
        
        # Header heuristics (stricter than before):
        # 1. Markdown header (# Title)
        # 2. Reasonable length (10-50 chars), no punctuation end
        # 3. Title Case or ALL CAPS with multiple words
        # 4. Not a table cell fragment
        
        if len(stripped) < 10 or len(stripped) > 60:
            current_content.append(line)
            continue
            
        if stripped.endswith(('.', ',', ':', ';', '?', '!', '|')):
            current_content.append(line)
            continue
        
        # Markdown header
        if stripped.startswith('#'):
            header_text = stripped.lstrip('#').strip()
            is_header = len(header_text) > 5
        # ALL CAPS (but not single words or acronyms)
        elif stripped.isupper() and ' ' in stripped and len(stripped.split()) >= 2:
            header_text = stripped.title()  # Convert to Title Case
            is_header = True
        # Title Case with 2-6 words
        elif stripped[0].isupper() and ' ' in stripped:
            words = stripped.split()
            if 2 <= len(words) <= 6:
                caps_words = sum(1 for w in words if w[0].isupper())
                # At least 70% words capitalized, not all lowercase after first
                if caps_words >= len(words) * 0.7:
                    # Extra check: not a sentence (no lowercase continuing words)
                    lowercase_midwords = sum(1 for w in words[1:] if w[0].islower() and len(w) > 3)
                    if lowercase_midwords == 0:
                        header_text = stripped
                        is_header = True
        
        if is_header and header_text:
            # Save previous section if it has meaningful content
            content_text = '\n'.join(current_content).strip()
            if content_text and len(content_text) > 30:
                sections.append((current_title, content_text))
            
            current_title = header_text
            current_content = []
        else:
            current_content.append(line)
    
    # Don't forget the last section
    content_text = '\n'.join(current_content).strip()
    if content_text and len(content_text) > 30:
        sections.append((current_title, content_text))
    
    return sections


def _chunk_section(
    section_title: str,
    section_content: str,
    max_chunk_tokens: int,
    similarity_threshold: float
) -> List[Tuple[str, str]]:
    """
    Chunk a single section, preserving the section title context.
    
    Returns:
        List of (chunk_text, section_title) tuples
    """
    tokens = estimate_tokens(section_content)
    
    # If section fits in one chunk, return as-is with title prefix
    if tokens <= max_chunk_tokens:
        chunk_with_context = f"[{section_title}]\n{section_content}"
        return [(chunk_with_context, section_title)]
    
    # Section too large - split by sentences using semantic chunking
    nlp = get_nlp()
    doc = nlp(section_content)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if not sentences:
        return [(f"[{section_title}]\n{section_content}", section_title)]
    
    # Use semantic chunking within the section
    chunks, _ = _semantic_chunk_with_sentences(
        sentences,
        max_chunk_tokens=max_chunk_tokens,
        similarity_threshold=similarity_threshold
    )
    
    # Prefix each chunk with section title for context
    return [(f"[{section_title}]\n{chunk}", section_title) for chunk in chunks]


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2 normalize each row of a matrix."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def get_embeddings_batch(sentences: List[str], batch_size: int = _DEFAULT_EMBED_BATCH) -> np.ndarray:
    """
    Get embeddings for a list of sentences using TRUE batch encoding.
    
    This uses the optimized get_embeddings_batch from embedding_local which
    processes all sentences in a single forward pass (6-7x faster than individual calls).
    
    Args:
        sentences: List of sentences to embed
        batch_size: Batch size for encoding
    
    Returns:
        (N, D) numpy array of L2-normalized vectors (cosine-ready)
    """
    if not sentences:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Use TRUE batch embedding from embedding_local
    embeddings_list = _get_embeddings_batch_from_local(list(sentences), batch_size=batch_size)
    
    if not embeddings_list:
        return np.zeros((0, 0), dtype=np.float32)
    
    embs_array = np.array(embeddings_list, dtype=np.float32)
    # Already normalized by sentence-transformers, but ensure consistency
    embs_array = _normalize_rows(embs_array)
    return embs_array


def _semantic_chunk_with_sentences(
    sentences: List[str],
    max_chunk_tokens: int = _DEFAULT_CHUNK_SIZE,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
) -> Tuple[List[str], List[List[str]]]:
    """
    Build semantically-coherent chunks and return both the chunk texts AND the sentence lists.
    
    This avoids re-parsing chunks later for overlap calculation.
    
    Returns:
        Tuple of (chunk_texts, chunk_sentence_lists)
    """
    if not sentences:
        return [], []

    sent_embs = get_embeddings_batch(sentences)
    
    if sent_embs.size == 0:
        # Fallback: chunk by naive token size if embeddings failed
        logger.warning("No embeddings returned; falling back to naive token-based sentence join.")
        chunks: List[str] = []
        chunk_sents_list: List[List[str]] = []
        cur: List[str] = []
        cur_toks = 0
        for s in sentences:
            s_toks = estimate_tokens(s)
            if cur and cur_toks + s_toks > max_chunk_tokens:
                chunks.append(" ".join(cur))
                chunk_sents_list.append(cur[:])
                cur = [s]
                cur_toks = s_toks
            else:
                cur.append(s)
                cur_toks += s_toks
        if cur:
            chunks.append(" ".join(cur))
            chunk_sents_list.append(cur[:])
        return chunks, chunk_sents_list

    chunks: List[str] = []
    chunk_sents_list: List[List[str]] = []
    current_chunk_sents: List[str] = []
    current_embs: List[np.ndarray] = []

    for i, sent in enumerate(sentences):
        emb = sent_embs[i]

        if not current_chunk_sents:
            current_chunk_sents.append(sent)
            current_embs.append(emb)
            continue

        # Centroid: mean then normalize
        centroid = np.mean(np.vstack(current_embs), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        sim = float(np.dot(emb, centroid))  # cosine similarity

        tentative_text = " ".join(current_chunk_sents + [sent])
        tentative_tokens = estimate_tokens(tentative_text)

        # Break conditions: token overflow OR (low similarity AND chunk not too small)
        if tentative_tokens > max_chunk_tokens or (sim < similarity_threshold and tentative_tokens > 50):
            chunks.append(" ".join(current_chunk_sents))
            chunk_sents_list.append(current_chunk_sents[:])
            current_chunk_sents = [sent]
            current_embs = [emb]
        else:
            current_chunk_sents.append(sent)
            current_embs.append(emb)

    if current_chunk_sents:
        chunks.append(" ".join(current_chunk_sents))
        chunk_sents_list.append(current_chunk_sents[:])

    return chunks, chunk_sents_list


def semantic_chunk_sentences(
    sentences: List[str],
    max_chunk_tokens: int = _DEFAULT_CHUNK_SIZE,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
) -> List[str]:
    """
    Build semantically-coherent chunks by iterating sentence-by-sentence.
    
    Legacy wrapper for backward compatibility.
    """
    chunks, _ = _semantic_chunk_with_sentences(sentences, max_chunk_tokens, similarity_threshold)
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP_SENTENCES,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    use_sections: bool = True
) -> List[str]:
    """
    Chunk text into semantic chunks with section awareness.
    
    Three modes:
    1. Section-aware (default for IT docs): Splits by headers first, then chunks within sections
    2. Short-line content (>60% lines < 50 chars): Line-based semantic chunking
    3. Normal prose: Sentence-based semantic chunking
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of sentences to overlap between chunks
        similarity_threshold: Cosine similarity threshold for semantic breaks
        use_sections: Whether to use section-aware chunking (default True)
    
    Returns:
        List of chunk strings
    """
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # Try section-aware chunking first (best for IT docs with headers)
    if use_sections:
        sections = _extract_sections(text)
        
        # If we found multiple sections, use section-aware chunking
        if len(sections) > 1:
            logger.debug(f"Using section-aware chunking: {len(sections)} sections found")
            all_chunks: List[str] = []
            
            for section_title, section_content in sections:
                section_chunks = _chunk_section(
                    section_title,
                    section_content,
                    max_chunk_tokens=chunk_size,
                    similarity_threshold=similarity_threshold
                )
                all_chunks.extend([chunk for chunk, _ in section_chunks])
            
            # Filter tiny chunks
            all_chunks = [c for c in all_chunks if len(c.strip()) > 30]
            
            if all_chunks:
                return all_chunks
            # Fall through to other methods if section chunking produced nothing

    # Short-line heuristic (e.g., code, tables, lists)
    # IMPORTANT: We still use semantic chunking, but treat each line as a "sentence"
    # This ensures we NEVER cut mid-word or mid-line
    short_line_ratio = sum(1 for line in lines if len(line) < 50) / len(lines)
    
    if short_line_ratio > 0.6:
        # Treat each line as a unit - chunk semantically by lines
        # This preserves whole lines and uses embedding similarity for grouping
        chunks, chunk_sents_list = _semantic_chunk_with_sentences(
            lines,  # Use lines as "sentences" 
            max_chunk_tokens=chunk_size,
            similarity_threshold=similarity_threshold
        )
        
        # Apply overlap (line-based)
        if chunk_overlap and chunk_overlap > 0 and len(chunks) > 1:
            final_chunks: List[str] = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_lines = chunk_sents_list[i - 1]
                overlap_n = min(len(prev_lines), chunk_overlap)
                overlap_text = " ".join(prev_lines[-overlap_n:]).strip()
                combined = f"{overlap_text} {chunks[i]}".strip() if overlap_text else chunks[i]
                final_chunks.append(combined)
            return final_chunks
        return chunks

    # Sentence-based semantic chunking (for normal prose)
    nlp = get_nlp()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Get chunks AND their sentence lists (avoids re-parsing)
    chunks, chunk_sents_list = _semantic_chunk_with_sentences(
        sentences,
        max_chunk_tokens=chunk_size,
        similarity_threshold=similarity_threshold
    )

    # Filter tiny chunks (and their sentence lists)
    filtered = [(c, s) for c, s in zip(chunks, chunk_sents_list) if len(c.strip()) > 10]
    if filtered:
        chunks = [f[0] for f in filtered]
        chunk_sents_list = [f[1] for f in filtered]
    else:
        return []

    # Apply overlap using cached sentence lists (NO RE-PARSING!)
    if chunk_overlap and chunk_overlap > 0 and len(chunks) > 1:
        final_chunks: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_sents = chunk_sents_list[i - 1]
            overlap_n = min(len(prev_sents), chunk_overlap)
            overlap_text = " ".join(prev_sents[-overlap_n:]).strip()
            combined = f"{overlap_text} {chunks[i]}".strip() if overlap_text else chunks[i]
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
