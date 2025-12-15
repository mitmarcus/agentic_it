from cremedelacreme import Node
from typing import Dict, List
from utils.logger import get_logger
from .config import _RAG_CONFIG

logger = get_logger(__name__)

# ============================================================================
# Offline Indexing Nodes
# ============================================================================

class LoadDocumentsNode(Node):
    """Load documents from source directory."""
    
    def prep(self, shared: Dict) -> str:
        """Get source directory from shared or cached config."""
        return shared.get("source_dir", _RAG_CONFIG["source_dir"])
    
    def exec(self, source_dir: str) -> List[Dict]:
        """Load all documents from directory using document parser."""
        from pathlib import Path
        from utils.document_parser import parse_document
        
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")
        
        # Find all supported files
        file_extensions = (".txt", ".md", ".html", ".htm", ".pdf")
        files = [f for f in source_path.rglob("*") if f.is_file() and f.suffix.lower() in file_extensions]
        
        logger.info(f"Found {len(files)} files in {source_path}")
        
        documents = []
        for filepath in files:
            try:
                relative_path = filepath.relative_to(source_path)
                
                # Use document parser for PDF and HTML, plain text for others
                file_ext = filepath.suffix.lower()
                if file_ext in [".pdf", ".html", ".htm"]:
                    result = parse_document(str(filepath))
                    content = result['text']
                else:
                    # Plain text/markdown
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                documents.append({
                    "content": content,
                    "metadata": {
                        "source_file": str(filepath),
                        "relative_path": str(relative_path),
                        "filename": filepath.name,
                        "extension": filepath.suffix,
                        "size_bytes": filepath.stat().st_size,
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} documents from {source_dir}")
        return documents
    
    def post(self, shared: Dict, prep_res: str, exec_res: List[Dict]) -> str:
        """Write documents to shared store."""
        shared["documents"] = exec_res
        return "default"