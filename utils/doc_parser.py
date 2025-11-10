"""
Document parsing utilities for multiple formats.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Any


def parse_document(file_path: str, doc_type: str = None) -> Dict[str, Any]:
    """
    Parse various document formats (PDF, MD, HTML, DOCX).
    
    Args:
        file_path: Path to document
        doc_type: Document type (auto-detected if None)
    
    Returns:
        Dict with title, sections, metadata
    """
    path = Path(file_path)
    
    if doc_type is None:
        doc_type = path.suffix.lower().strip(".")
    
    # Compute document hash for deduplication
    with open(file_path, "rb") as f:
        doc_hash = hashlib.md5(f.read()).hexdigest()
    
    # Parse based on type
    if doc_type == "md" or doc_type == "markdown":
        return _parse_markdown(path, doc_hash)
    elif doc_type == "pdf":
        return _parse_pdf(path, doc_hash)
    elif doc_type == "html":
        return _parse_html(path, doc_hash)
    elif doc_type == "docx":
        return _parse_docx(path, doc_hash)
    else:
        # Treat as plain text
        return _parse_text(path, doc_hash)


def _parse_markdown(path: Path, doc_hash: str) -> Dict[str, Any]:
    """Parse Markdown document."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract title (first # heading)
    lines = content.split("\n")
    title = path.stem
    for line in lines:
        if line.startswith("# "):
            title = line.strip("# ").strip()
            break
    
    # Simple section splitting (by ## headings)
    sections = []
    current_heading = "Introduction"
    current_content = []
    
    for line in lines:
        if line.startswith("## "):
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content).strip()
                })
            current_heading = line.strip("#").strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Add last section
    if current_content:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_content).strip()
        })
    
    return {
        "title": title,
        "sections": sections,
        "metadata": {
            "source": str(path),
            "category": "documentation",
            "doc_type": "markdown"
        },
        "hash": doc_hash
    }


def _parse_pdf(path: Path, doc_hash: str) -> Dict[str, Any]:
    """Parse PDF document.
    
    Raises:
        Exception: If PDF parsing fails (let Node retry mechanism handle it)
    """
    from PyPDF2 import PdfReader
    
    reader = PdfReader(str(path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return {
        "title": path.stem,
        "sections": [{"heading": "Content", "content": text}],
        "metadata": {
            "source": str(path),
            "category": "documentation",
            "doc_type": "pdf",
            "pages": len(reader.pages)
        },
        "hash": doc_hash
    }


def _parse_html(path: Path, doc_hash: str) -> Dict[str, Any]:
    """Parse HTML document.
    
    Raises:
        Exception: If HTML parsing fails (let Node retry mechanism handle it)
    """
    from bs4 import BeautifulSoup
    
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    # Extract title
    title = soup.title.string if soup.title else path.stem
    
    # Remove script/style
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    # Extract text
    text = soup.get_text(separator="\n", strip=True)
    
    return {
        "title": title,
        "sections": [{"heading": "Content", "content": text}],
        "metadata": {
            "source": str(path),
            "category": "documentation",
            "doc_type": "html"
        },
        "hash": doc_hash
    }


def _parse_docx(path: Path, doc_hash: str) -> Dict[str, Any]:
    """Parse DOCX document.
    
    Raises:
        Exception: If DOCX parsing fails (let Node retry mechanism handle it)
    """
    from docx import Document
    
    doc = Document(str(path))
    text = "\n".join([para.text for para in doc.paragraphs])
    
    return {
        "title": path.stem,
        "sections": [{"heading": "Content", "content": text}],
        "metadata": {
            "source": str(path),
            "category": "documentation",
            "doc_type": "docx"
        },
        "hash": doc_hash
    }


def _parse_text(path: Path, doc_hash: str) -> Dict[str, Any]:
    """Parse plain text document.
    
    Raises:
        Exception: If file reading fails (let Node retry mechanism handle it)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return {
        "title": path.stem,
        "sections": [{"heading": "Content", "content": text}],
        "metadata": {
            "source": str(path),
            "category": "documentation",
            "doc_type": "text"
        },
        "hash": doc_hash
    }


if __name__ == "__main__":
    # Test with a sample markdown file
    try:
        import sys
        if len(sys.argv) > 1:
            path = sys.argv[1]
            result = parse_document(path)
            print(f"Title: {result['title']}")
            print(f"Sections: {len(result['sections'])}")
            print(f"Metadata: {result['metadata']}")
    except Exception as e:
        print(f"\nError during standalone test: {e}")
        print("\nNote: In production, Nodes handle retries and fallbacks.")
        print("This test block catches errors for standalone execution only.")
