"""
ChromaDB client for vector database operations.
"""
import os
from typing import Any, Sequence
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection


# Default configuration
_DEFAULT_COLLECTION_NAME = "it_support_docs"
_DEFAULT_MODE = "client"
_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 8000
_DEFAULT_PERSIST_DIR = "./chroma_data"

# Cached instances 
_CLIENT_CACHE: dict[str, Any] = {}
_COLLECTION_CACHE: dict[str, Collection] = {}


def initialize_client(
    *,
    mode: str = os.getenv("CHROMADB_MODE", _DEFAULT_MODE),
    host: str = os.getenv("CHROMADB_HOST", _DEFAULT_HOST),
    port: int = int(os.getenv("CHROMADB_PORT", str(_DEFAULT_PORT))),
    persist_dir: str = os.getenv("CHROMADB_PERSIST_DIR", _DEFAULT_PERSIST_DIR)
) -> Any:
    """
    Initialize ChromaDB client based on environment configuration.
    
    Args:
        mode: Client mode - 'server' or 'client' (defaults to env var or 'client')
        host: Server host for server mode (defaults to env var or 'localhost')
        port: Server port for server mode (defaults to env var or 8000)
        persist_dir: Directory for persistent client (defaults to env var or './chroma_data')
    
    Returns:
        ChromaDB client instance
        
    Raises:
        Exception: If client initialization fails (let Node retry)
        
    Example:
        >>> client = initialize_client(mode="server", host="localhost", port=8001)
    """
    # Create cache key based on configuration
    cache_key = f"{mode}:{host}:{port}:{persist_dir}"
    
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
    
    if mode == "server":
        # Connect to remote ChromaDB server (Docker setup)
        print(f"Connecting to ChromaDB server at {host}:{port}")
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False)
        )
    else:
        # Use persistent local client
        print(f"Using local ChromaDB at {persist_dir}")
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
    
    _CLIENT_CACHE[cache_key] = client
    return client


def get_collection(
    *,
    collection_name: str = os.getenv("CHROMADB_COLLECTION", _DEFAULT_COLLECTION_NAME)
) -> Collection:
    """
    Get or create ChromaDB collection.
    
    Args:
        collection_name: Name of collection (defaults to env var or 'it_support_docs')
    
    Returns:
        Collection instance
        
    Raises:
        Exception: If collection access fails (let Node retry)
        
    Example:
        >>> collection = get_collection(collection_name="my_docs")
    """
    # Return cached collection if available
    if collection_name in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[collection_name]
    
    client = initialize_client()
    
    # Get or create collection with cosine similarity
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "IT support documentation and knowledge base",
            "hnsw:space": "cosine"
        } 
    )
    
    _COLLECTION_CACHE[collection_name] = collection
    print(f"Using collection: {collection_name} ({collection.count()} documents)")
    
    return collection


def insert_documents(
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadata: Sequence[dict[str, Any]],
    *,
    ids: tuple[str, ...] = (),
    collection_name: str = os.getenv("CHROMADB_COLLECTION", _DEFAULT_COLLECTION_NAME)
) -> None:
    """
    Insert documents into ChromaDB collection.
    
    Args:
        chunks: Sequence of text chunks
        embeddings: Sequence of embedding vectors
        metadata: Sequence of metadata dicts for each chunk
        ids: Document IDs (auto-generated if empty tuple)
        collection_name: Collection name (defaults to env var or 'it_support_docs')
    
    Raises:
        Exception: If insertion fails (let Node retry)
        
    Example:
        >>> insert_documents(
        ...     chunks=["text1", "text2"],
        ...     embeddings=[[0.1, 0.2], [0.3, 0.4]],
        ...     metadata=[{"source": "doc1"}, {"source": "doc2"}]
        ... )
    """
    collection = get_collection(collection_name=collection_name)
    
    # Generate IDs if not provided
    if not ids:
        import hashlib
        ids = tuple(
            hashlib.md5(f"{chunk[:100]}{i}".encode()).hexdigest()
            for i, chunk in enumerate(chunks)
        )
    
    # Insert into collection (convert to lists for ChromaDB API)
    collection.add(
        documents=list(chunks),
        embeddings=[list(emb) for emb in embeddings],
        metadatas=list(metadata),
        ids=list(ids)
    )
    
    print(f"Inserted {len(chunks)} documents into {collection.name}")


def query_collection(
    query_embedding: Sequence[float],
    *,
    top_k: int = 3,
    filter_metadata: dict[str, Any] | None = None,
    collection_name: str = os.getenv("CHROMADB_COLLECTION", _DEFAULT_COLLECTION_NAME)
) -> list[dict[str, Any]]:
    """
    Query ChromaDB collection for similar documents.
    
    Args:
        query_embedding: Query vector
        top_k: Number of results to return (default: 3)
        filter_metadata: Metadata filter (e.g., {"category": "networking"})
        collection_name: Collection name (defaults to env var or 'it_support_docs')
    
    Returns:
        List of result dicts with keys: id, document, metadata, distance, score
        
    Raises:
        Exception: If query fails (let Node retry)
        
    Example:
        >>> results = query_collection(
        ...     [0.1, 0.2, 0.3],
        ...     top_k=5,
        ...     filter_metadata={"category": "networking"}
        ... )
    """
    collection = get_collection(collection_name=collection_name)
    
    # Query collection
    results = collection.query(
        query_embeddings=[list(query_embedding)],
        n_results=top_k,
        where=filter_metadata,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results (safely access nested structures)
    formatted_results = []
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    
    for i in range(len(ids)):
        distance = distances[i]
        # For cosine distance: range is [0, 2], where 0 = identical, 2 = opposite
        # Convert to similarity score: 1 - (distance / 2) = [0, 1] range
        similarity_score = 1.0 - (distance / 2.0)
        
        formatted_results.append({
            "id": ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "distance": distance,
            "score": similarity_score
        })
    
    return formatted_results


def get_neighbor_chunks(
    source_file: str,
    chunk_index: int,
    neighbor_count: int = 1,
    *,
    collection_name: str = os.getenv("CHROMADB_COLLECTION", _DEFAULT_COLLECTION_NAME)
) -> list[dict[str, Any]]:
    """
    Fetch neighboring chunks from the same document.
    
    This is useful when a chunk is highly relevant but the answer spans multiple chunks
    (e.g., step-by-step instructions split across chunks).
    
    Args:
        source_file: Source file path
        chunk_index: Index of the anchor chunk
        neighbor_count: How many neighbors to fetch on each side (default: 1)
        collection_name: Collection name
    
    Returns:
        List of neighbor chunks sorted by chunk_index
        
    Example:
        >>> # Get chunk 5 and its neighbors (4, 6)
        >>> neighbors = get_neighbor_chunks("guide.html", 5, neighbor_count=1)
    """
    collection = get_collection(collection_name=collection_name)
    
    # Calculate neighbor indices
    min_index = max(0, chunk_index - neighbor_count)
    max_index = chunk_index + neighbor_count
    
    # Fetch all chunks from this source
    results = collection.get(
        where={"source_file": source_file},
        include=["documents", "metadatas"]
    )
    
    if not results or not results.get("ids"):
        return []
    
    # Filter and format neighbors
    neighbors = []
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    
    for i in range(len(ids)):
        metadata = metadatas[i] if i < len(metadatas) else {}
        current_index = metadata.get("chunk_index", -1)
        
        # Include if in neighbor range
        if min_index <= current_index <= max_index:
            neighbors.append({
                "id": ids[i],
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadata,
                "score": 0.95,  # High score for neighbors
                "is_neighbor": True
            })
    
    # Sort by chunk_index
    neighbors.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
    return neighbors


def delete_documents_by_source(
    source_file: str,
    *,
    collection_name: str = os.getenv("CHROMADB_COLLECTION", _DEFAULT_COLLECTION_NAME)
) -> int:
    """
    Delete all documents from a specific source file.
    
    Args:
        source_file: Source file path to delete
        collection_name: Collection name (defaults to env var or 'it_support_docs')
    
    Returns:
        Number of documents deleted
        
    Example:
        >>> deleted = delete_documents_by_source("./data/docs/guide.md")
    """
    collection = get_collection(collection_name=collection_name)
    
    # Get all documents with this source file
    results = collection.get(
        where={"source_file": source_file},
        include=["documents"]
    )
    
    ids_to_delete = results.get("ids", [])
    
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} documents from source: {source_file}")
    
    return len(ids_to_delete)


def delete_collection(
    *,
    collection_name: str = os.getenv("CHROMADB_COLLECTION", _DEFAULT_COLLECTION_NAME)
) -> None:
    """
    Delete a collection (use with caution!).
    
    Args:
        collection_name: Collection name (defaults to env var or 'it_support_docs')
        
    Example:
        >>> delete_collection(collection_name="test_collection")
    """
    client = initialize_client()
    client.delete_collection(name=collection_name)
    
    # Remove from cache if present
    _COLLECTION_CACHE.pop(collection_name, None)
    
    print(f"Deleted collection: {collection_name}")


def get_collection_stats(
    *,
    collection_name: str = os.getenv("CHROMADB_COLLECTION", _DEFAULT_COLLECTION_NAME)
) -> dict[str, Any]:
    """
    Get statistics about a collection.
    
    Args:
        collection_name: Collection name (defaults to env var or 'it_support_docs')
    
    Returns:
        Dict with collection statistics (name, count, metadata)
        
    Example:
        >>> stats = get_collection_stats(collection_name="my_docs")
    """
    collection = get_collection(collection_name=collection_name)
    
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata
    }


if __name__ == "__main__":
    # Test ChromaDB client
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    # When running from host machine, use localhost:8001
    # Inside Docker, the chatbot service uses chromadb:8000
    os.environ["CHROMADB_HOST"] = "localhost"
    os.environ["CHROMADB_PORT"] = "8001"
    
    print("Testing ChromaDB client...")
    print("Ensure ChromaDB server is running: docker compose up -d chromadb")
    print()
    
    try:
        # Initialize client
        client = initialize_client()
        print(f"✓ Client initialized: {type(client)}")
        
        # Get collection
        collection = get_collection(collection_name="test_collection")
        print(f"✓ Collection created: {collection.name}")
        
        # Insert test documents
        test_chunks = [
            "VPN troubleshooting guide: Check your internet connection first.",
            "Printer setup: Connect via USB or network configuration.",
            "Password reset: Use the self-service portal at portal.company.com"
        ]
        
        from embedding_local import get_embedding
        test_embeddings = [get_embedding(chunk) for chunk in test_chunks]
        test_metadata = [
            {"source": "vpn_guide.md", "category": "networking"},
            {"source": "printer_guide.md", "category": "hardware"},
            {"source": "password_guide.md", "category": "access"}
        ]
        
        insert_documents(
            test_chunks, 
            test_embeddings, 
            test_metadata, 
            collection_name="test_collection"
        )
        print(f"✓ Documents inserted")
        
        # Query collection
        query_text = "How do I fix VPN issues?"
        query_emb = get_embedding(query_text)
        results = query_collection(
            query_emb, 
            top_k=2, 
            collection_name="test_collection"
        )
        
        print(f"\n✓ Query results for: '{query_text}'")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.3f}")
            print(f"     Document: {result['document'][:60]}...")
            print(f"     Metadata: {result['metadata']}")
        
        # Get stats
        stats = get_collection_stats(collection_name="test_collection")
        print(f"\n✓ Collection stats: {stats}")
        
        # Cleanup
        delete_collection(collection_name="test_collection")
        print(f"\n✓ Test collection deleted")
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
