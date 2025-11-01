"""
Flows for IT Support Chatbot.

Flows orchestrate nodes into complete workflows.
"""
import logging
from langfuse_tracing import trace_flow, TracingConfig

from cremedelacreme import Flow
from nodes import (
    # Online query nodes
    IntentClassificationNode,
    EmbedQueryNode,
    SearchKnowledgeBaseNode,
    DecisionMakerNode,
    GenerateAnswerNode,
    AskClarifyingQuestionNode,
    FormatFinalResponseNode,
    InteractiveTroubleshootNode,
    NotImplementedNode,
    # Offline indexing nodes
    LoadDocumentsNode,
    ChunkDocumentsNode,
    EmbedDocumentsNode,
    StoreInChromaDBNode,
)

logger = logging.getLogger(__name__)

# Initialize tracing configuration
try:
    tracing_config = TracingConfig.from_env()
    logger.info("Tracing configuration loaded from environment")
except Exception as e:
    logger.warning(f"Failed to load tracing configuration: {e}")
    tracing_config = None


# ============================================================================
# Main Query Flow
# ============================================================================

@trace_flow(config=tracing_config, flow_name="ITSupportQueryFlow")
class QueryFlow(Flow):
    """Main query answering flow with tracing."""
    def __init__(self):
        # Create nodes
        intent_node = IntentClassificationNode()
        embed_node = EmbedQueryNode()
        search_node = SearchKnowledgeBaseNode()
        decision_node = DecisionMakerNode()
        answer_node = GenerateAnswerNode()
        clarify_node = AskClarifyingQuestionNode()
        format_node = FormatFinalResponseNode()
        
        # Interactive troubleshooting node
        troubleshoot_node = InteractiveTroubleshootNode()
        
        # Placeholder nodes for not-yet-implemented features
        search_tickets_node = NotImplementedNode("Ticket search")
        create_ticket_node = NotImplementedNode("Ticket creation")
        
        # Connect nodes
        # Linear path: intent -> embed -> search
        _ = intent_node >> embed_node >> search_node
        
        # Search can go to decision maker regardless of whether docs found
        _ = search_node - "docs_found" >> decision_node
        _ = search_node - "no_docs" >> decision_node
        
        # Decision maker routes to different actions
        _ = decision_node - "answer" >> answer_node
        _ = decision_node - "clarify" >> clarify_node
        _ = decision_node - "search_kb" >> embed_node  # Loop back to search again
        _ = decision_node - "troubleshoot" >> troubleshoot_node
        _ = decision_node - "search_tickets" >> search_tickets_node
        _ = decision_node - "create_ticket" >> create_ticket_node
        
        # All paths lead to format response
        _ = answer_node >> format_node
        _ = clarify_node >> format_node
        _ = troubleshoot_node >> format_node
        _ = search_tickets_node >> format_node
        _ = create_ticket_node >> format_node
        
        # Initialize Flow with start node
        super().__init__(start=intent_node)
        
        logger.info("Query flow created with tracing")


def create_query_flow() -> Flow:
    """
    Create the main query answering flow.
    
    Flow:
    1. Classify intent
    2. Embed query
    3. Search knowledge base
    4. Decision maker (agent) routes to:
       - answer: Generate answer
       - clarify: Ask for more info
       - search_kb: Re-search (loops back to search)
       - troubleshoot/search_tickets/create_ticket: Not yet implemented
    5. Format final response
    
    Returns:
        Configured Flow instance with tracing
    """
    return QueryFlow()


# ============================================================================
# Offline Indexing Flow
# ============================================================================

@trace_flow(config=tracing_config, flow_name="ITSupportIndexingFlow")
class IndexingFlow(Flow):
    """Offline document indexing flow with tracing."""
    def __init__(self):
        # Create nodes
        load_node = LoadDocumentsNode()
        chunk_node = ChunkDocumentsNode()
        embed_node = EmbedDocumentsNode()
        store_node = StoreInChromaDBNode()
        
        # Connect in sequence
        _ = load_node >> chunk_node >> embed_node >> store_node
        
        # Initialize Flow with start node
        super().__init__(start=load_node)
        
        logger.info("Indexing flow created with tracing")


def create_indexing_flow() -> Flow:
    """
    Create the offline document indexing flow.
    
    Flow:
    1. Load documents from directory
    2. Chunk documents (batch processing)
    3. Embed chunks (batch processing)
    4. Store in ChromaDB
    
    Returns:
        Configured Flow instance with tracing
    """
    return IndexingFlow()


# ============================================================================
# Simplified Query Flow (for testing/development)
# ============================================================================

@trace_flow(config=tracing_config, flow_name="ITSupportSimpleQueryFlow")
class SimpleQueryFlow(Flow):
    """Simplified query flow for testing with tracing."""
    def __init__(self):
        # Create nodes
        intent_node = IntentClassificationNode()
        embed_node = EmbedQueryNode()
        search_node = SearchKnowledgeBaseNode()
        answer_node = GenerateAnswerNode()
        format_node = FormatFinalResponseNode()
        
        # Simple linear flow
        _ = intent_node >> embed_node >> search_node
        
        # Both search outcomes go to answer
        _ = search_node - "docs_found" >> answer_node
        _ = search_node - "no_docs" >> answer_node
        
        _ = answer_node >> format_node
        
        # Initialize Flow with start node
        super().__init__(start=intent_node)
        
        logger.info("Simple query flow created with tracing")


def create_simple_query_flow() -> Flow:
    """
    Useful for testing RAG pipeline without agent routing.
    Goes straight from search to answer.
    
    Returns:
        Configured Flow instance with tracing
    """
    return SimpleQueryFlow()


# ============================================================================
# Flow Factory
# ============================================================================

def get_flow(flow_type: str = "query") -> Flow:
    """
    Get a flow by type name.
    
    Args:
        flow_type: One of "query", "indexing", "simple"
    
    Returns:
        Flow instance
        
    Raises:
        ValueError: If flow_type is unknown
    """
    flows = {
        "query": create_query_flow,
        "indexing": create_indexing_flow,
        "simple": create_simple_query_flow
    }
    
    if flow_type not in flows:
        raise ValueError(f"Unknown flow type: {flow_type}. Available: {list(flows.keys())}")
    
    return flows[flow_type]()


if __name__ == "__main__":
    # Test flow creation
    print("Testing flow creation...")
    
    try:
        query_flow = create_query_flow()
        print(f"✓ Query flow created: {query_flow}")
        
        indexing_flow = create_indexing_flow()
        print(f"✓ Indexing flow created: {indexing_flow}")
        
        simple_flow = create_simple_query_flow()
        print(f"✓ Simple query flow created: {simple_flow}")
        
        print("\n✓ All flows created successfully")
        
    except Exception as e:
        print(f"\n✗ Error creating flows: {e}")
        import traceback
        traceback.print_exc()
