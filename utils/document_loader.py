"""
Document loader for various file formats.
"""
import os
from typing import List, Dict, Any
from pathlib import Path


def load_text_file(filepath: str) -> str:
    """
    Load plain text file.
    
    Args:
        filepath: Path to text file
    
    Returns:
        File contents as string
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_markdown_file(filepath: str) -> str:
    """
    Load markdown file (same as text for now).
    
    Args:
        filepath: Path to markdown file
    
    Returns:
        File contents as string
    """
    return load_text_file(filepath)


def load_documents_from_directory(
    source_dir: str | None = None,
    *,
    file_extensions: tuple[str, ...] = (".txt", ".md"),
    recursive: bool = True
) -> List[Dict[str, Any]]:
    """
    Load all documents from directory.
    
    Args:
        source_dir: Source directory path (defaults to INGESTION_SOURCE_DIR env var)
        file_extensions: File extensions to include (default: .txt, .md)
        recursive: Whether to search subdirectories
    
    Returns:
        List of document dicts with 'content', 'metadata' keys
        
    Raises:
        FileNotFoundError: If source directory doesn't exist
    """
    source_path = Path(source_dir or os.getenv("INGESTION_SOURCE_DIR", "./data/docs"))
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    
    # Find matching files
    pattern = source_path.rglob("*") if recursive else source_path.glob("*")
    files = [f for f in pattern if f.is_file() and f.suffix in file_extensions]
    
    print(f"Found {len(files)} files in {source_path}")
    
    # Load each file
    documents = []
    for filepath in files:
        try:
            relative_path = filepath.relative_to(source_path)
            
            documents.append({
                "content": load_text_file(str(filepath)),
                "metadata": {
                    "source_file": str(filepath),
                    "relative_path": str(relative_path),
                    "filename": filepath.name,
                    "extension": filepath.suffix,
                    "size_bytes": filepath.stat().st_size,
                    "doc_type": infer_doc_type(filepath.name),
                    "category": infer_category(str(relative_path))
                }
            })
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            continue
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents


def infer_doc_type(filename: str) -> str:
    """
    Infer document type from filename.
    
    Args:
        filename: Name of file
    
    Returns:
        Document type: guide, faq, troubleshooting, policy, other
    """
    filename_lower = filename.lower()
    
    if "guide" in filename_lower or "tutorial" in filename_lower:
        return "guide"
    elif "faq" in filename_lower or "question" in filename_lower:
        return "faq"
    elif "troubleshoot" in filename_lower or "debug" in filename_lower or "fix" in filename_lower:
        return "troubleshooting"
    elif "policy" in filename_lower or "procedure" in filename_lower:
        return "policy"
    else:
        return "other"


def infer_category(path: str) -> str:
    """
    Infer category from file path.
    
    Args:
        path: Relative file path
    
    Returns:
        Category: networking, hardware, software, access, security, other
    """
    path_lower = path.lower()
    
    if any(word in path_lower for word in ["network", "vpn", "wifi", "connection"]):
        return "networking"
    elif any(word in path_lower for word in ["hardware", "printer", "monitor", "laptop"]):
        return "hardware"
    elif any(word in path_lower for word in ["software", "application", "app", "program"]):
        return "software"
    elif any(word in path_lower for word in ["access", "password", "login", "auth"]):
        return "access"
    elif any(word in path_lower for word in ["security", "firewall", "antivirus", "malware"]):
        return "security"
    else:
        return "other"


if __name__ == "__main__":
    # Test document loader
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing document loader...")
    
    # Create test directory and files
    test_dir = "./test_docs"
    os.makedirs(test_dir, exist_ok=True)
    
    test_files = {
        "vpn_guide.md": """# VPN Setup Guide
        
This guide explains how to set up VPN on your device.

## Prerequisites
- Company laptop
- Active directory credentials

## Steps
1. Download Cisco AnyConnect
2. Connect to vpn.company.com
3. Enter your credentials
""",
        "printer_troubleshooting.txt": """Printer Troubleshooting

If your printer is not working:
1. Check power cable
2. Check USB/network connection
3. Restart printer
4. Reinstall drivers
5. Contact IT support if issue persists
""",
        "password_faq.md": """# Password FAQ

Q: How do I reset my password?
A: Use the self-service portal at portal.company.com

Q: How long should my password be?
A: Minimum 12 characters with mixed case, numbers, and symbols.
"""
    }
    
    # Write test files
    for filename, content in test_files.items():
        with open(os.path.join(test_dir, filename), 'w') as f:
            f.write(content)
    
    print(f"\n✓ Created test files in {test_dir}")
    
    # Load documents
    try:
        docs = load_documents_from_directory(test_dir, recursive=False)
        
        print(f"\n✓ Loaded {len(docs)} documents:")
        for doc in docs:
            print(f"\n  File: {doc['metadata']['filename']}")
            print(f"  Type: {doc['metadata']['doc_type']}")
            print(f"  Category: {doc['metadata']['category']}")
            print(f"  Size: {doc['metadata']['size_bytes']} bytes")
            print(f"  Content preview: {doc['content'][:80]}...")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n✓ Cleaned up test directory")
