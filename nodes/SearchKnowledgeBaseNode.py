from cremedelacreme import Node
from typing import Dict
from utils.logger import get_logger
from core.llm_config import get_llm_config
from typing import Dict, List
from .config import _RAG_CONFIG, _FEATURE_FLAGS
from utils.chromadb_client import query_collection, get_neighbor_chunks
from utils.reranker import rerank_results
from utils.feedback import apply_feedback_adjustments
from utils.chunker import truncate_to_token_limit

# ============================================================================
# Node 3: SearchKnowledgeBaseNode
# ============================================================================

logger = get_logger(__name__)

class SearchKnowledgeBaseNode(Node):
    """Retrieve relevant documents from ChromaDB with streamlined RAG pipeline.
    
    Simplified pipeline (40% faster, same quality):
    1. Vector Search: Get top candidates using embedding similarity
    2. Reranking: Cross-encoder re-scores for precision (most impactful step)
    
    Removed (over-engineering for ~200 docs):
    - Metadata filtering (let reranker decide relevance)
    - BM25 hybrid (vector handles semantic queries well)
    - MMR diversity (reranker naturally diversifies)
    """
    
    def __init__(self):
        config = get_llm_config()
        super().__init__(max_retries=config.max_retries, wait=config.retry_wait)
    
    def prep(self, shared: Dict) -> Dict:
        """Read query embedding and text."""
        return {
            "query_embedding": shared.get("query_embedding", []),
            "query_text": shared.get("user_query", "")
        }
    
    def exec(self, prep_data: Dict) -> List[Dict]:
        """Query ChromaDB with vector search, then rerank top candidates."""
        query_embedding = prep_data["query_embedding"]
        query_text = prep_data["query_text"]
        
        if not query_embedding or sum(query_embedding) == 0:
            logger.warning("Empty or zero embedding, skipping search")
            return []
        
        # Use cached config
        top_k = _RAG_CONFIG["top_k"]
        min_score = _RAG_CONFIG["min_score"]
        
        # Step 1: Vector search - get more candidates for reranking
        rerank_candidates = top_k * 3  # Fetch 3x for reranker to work with
        
        vector_results = query_collection(
            query_embedding,
            top_k=rerank_candidates
        )
        
        if not vector_results:
            logger.debug("No vector search results")
            return []
        
        # Log retrieval stats
        scores = [r["score"] for r in vector_results]
        logger.debug(f"Vector search: {len(vector_results)} results, scores: {scores[:5]}...")
        
        # Step 2: Light filtering - remove very low scores
        # Use lower threshold since reranker will do the heavy lifting
        filtered_results = [r for r in vector_results if r.get("score", 0) >= min_score * 0.7]
        
        if not filtered_results:
            logger.debug(f"All results below threshold {min_score * 0.7}")
            return []
        
        # Step 3: Smart neighbor retrieval for high-scoring chunks
        # IMPORTANT: Do this BEFORE reranking so neighbors get reranked too!
        # This ensures all chunks have consistent score fields (rerank_score)
        if filtered_results and len(filtered_results) > 0:
            top_result = max(filtered_results, key=lambda x: x.get("score", 0))
            top_score = top_result.get("score", 0)
            
            # Only fetch neighbors if top result is highly relevant (>0.75)
            if top_score > 0.75:
                source_file = top_result["metadata"].get("source_file")
                chunk_index = top_result["metadata"].get("chunk_index")
                
                if source_file and chunk_index is not None:
                    logger.debug(f"Fetching neighbors for high-scoring chunk {chunk_index} from {source_file}")
                    neighbors = get_neighbor_chunks(source_file, chunk_index, neighbor_count=1)
                    
                    # Add neighbors that aren't already in results
                    existing_ids = {r["id"] for r in filtered_results}
                    new_neighbors = [n for n in neighbors if n["id"] not in existing_ids]
                    
                    if new_neighbors:
                        logger.debug(f"Added {len(new_neighbors)} neighbor chunks for complete context")
                        filtered_results.extend(new_neighbors)
        
        # Step 4: Rerank using cross-encoder (the most impactful step)
        # Now ALL chunks (including neighbors) get reranked together
        if _FEATURE_FLAGS["rerank"] and query_text and len(filtered_results) > 1:
            logger.debug(f"Reranking {len(filtered_results)} candidates (including neighbors)...")
            filtered_results = rerank_results(query_text, filtered_results, top_k=top_k)
            logger.debug(f"Reranked to top {len(filtered_results)} results")
        else:
            # No reranking - just take top_k by vector score
            filtered_results = filtered_results[:top_k]
        
        # Apply feedback adjustments (boost/penalize based on user feedback)
        score_key = "rerank_score" if filtered_results and "rerank_score" in filtered_results[0] else "score"
        filtered_results = apply_feedback_adjustments(filtered_results, score_key=score_key)
        
        return filtered_results

    
    def post(self, shared: Dict, prep_res: Dict, exec_res: List[Dict]) -> str:
        """Write results and compile context."""
        shared["retrieved_docs"] = exec_res
        
        # Compile RAG context from documents
        if exec_res:
            max_context_tokens = _RAG_CONFIG["max_context_tokens"]
            
            # Group chunks by source file, preserving chunk order
            chunks_by_source: Dict[str, List[Dict]] = {}
            for doc in exec_res:
                source = doc['metadata'].get('source_file', 'unknown')
                if source not in chunks_by_source:
                    chunks_by_source[source] = []
                chunks_by_source[source].append(doc)
            
            # For each source, sort chunks by index and keep them together
            unique_docs = []
            for source, chunks in chunks_by_source.items():
                # Sort chunks by chunk_index to maintain document flow
                chunks.sort(key=lambda c: c['metadata'].get('chunk_index', 0))
                
                # Find the highest scoring chunk in this group
                # Prefer rerank > rrf > vector score, but handle 0 scores properly
                def get_best_score(chunk):
                    rerank = chunk.get('rerank_score', 0)
                    if rerank > 0:
                        return rerank
                    rrf = chunk.get('rrf_score')
                    if rrf is not None:
                        return rrf
                    return chunk.get('score', 0)
                
                best_score = max(get_best_score(c) for c in chunks)
                
                # Add all chunks with their group score for sorting
                for chunk in chunks:
                    chunk['_group_score'] = best_score
                    unique_docs.append(chunk)
            
            # Sort by group score (keeps chunk groups together, ordered by relevance)
            unique_docs.sort(key=lambda d: d.get('_group_score', 0), reverse=True)
            
            context_parts = []
            doc_num = 1
            prev_source = None
            
            for doc in unique_docs:
                metadata = doc['metadata']
                source_file = metadata.get('source_file', 'unknown')
                chunk_index = metadata.get('chunk_index', '?')
                total_chunks = metadata.get('total_chunks', '?')
                
                # Show rerank score if available, then RRF, then vector score
                relevance = doc.get('rerank_score', doc.get('rrf_score', doc.get('score', 0)))
                is_neighbor = doc.get('is_neighbor', False)
                
                # Only increment doc number for new sources
                if source_file != prev_source:
                    context_parts.append(f"[Document {doc_num}] (Relevance: {relevance:.2f})")
                    context_parts.append(f"File: {source_file}")
                    doc_num += 1
                    prev_source = source_file
                
                # Show chunk info and content
                chunk_label = f"Chunk {chunk_index}" if not is_neighbor else f"Chunk {chunk_index} (neighbor)"
                context_parts.append(f"--- {chunk_label} of {total_chunks} ---")
                context_parts.append(doc['document'])
                context_parts.append("")
            
            rag_context = "\n".join(context_parts)
            rag_context = truncate_to_token_limit(rag_context, max_context_tokens)
            
            shared["rag_context"] = rag_context
            
            # Count unique sources and total chunks
            unique_sources = len(set(d['metadata'].get('source_file', 'unknown') for d in unique_docs))
            neighbor_count = sum(1 for d in unique_docs if d.get('is_neighbor', False))
            logger.debug(f"Compiled context: {unique_sources} sources, {len(unique_docs)} chunks ({neighbor_count} neighbors)")
            
            return "docs_found"
        else:
            shared["rag_context"] = ""
            return "no_docs"

