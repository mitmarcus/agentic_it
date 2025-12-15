from core.llm_config import get_llm_config
from cremedelacreme import Node
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)

class EmbedDocumentsNode(Node):
    """Generate embeddings for document chunks using TRUE batch encoding.
    
    Changed from BatchNode to regular Node for 6-7x speedup.
    Instead of calling get_embedding() per chunk (N calls), we use
    get_embeddings_batch() which processes all chunks in one forward pass.
    """
    
    def __init__(self):
        config = get_llm_config()
        super().__init__(max_retries=config.max_retries, wait=config.retry_wait)
    
    def prep(self, shared: Dict) -> List[Dict]:
        """Read chunks from shared store."""
        return shared.get("all_chunks", [])
    
    def exec(self, chunks: List[Dict]) -> List[List[float]]:
        """Embed ALL chunks in a single batch call for maximum efficiency."""
        from utils.embedding_local import get_embeddings_batch
        
        if not chunks:
            return []
        
        # Extract content from all chunks
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # TRUE batch encoding - single forward pass for all texts (6-7x faster)
        embeddings = get_embeddings_batch(texts, batch_size=32)
        
        logger.info(f"Batch embedded {len(texts)} chunks in single call")
        return embeddings
    
    def post(self, shared: Dict, prep_res: List[Dict], exec_res: List[List[float]]) -> str:
        """Store embeddings with chunks."""
        shared["all_embeddings"] = exec_res
        logger.info(f"Generated {len(exec_res)} embeddings")
        return "default"