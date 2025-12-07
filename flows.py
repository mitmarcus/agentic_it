"""
Flows for IT Support Chatbot.

Flows orchestrate nodes into complete workflows.
"""
from langfuse_tracing import trace_flow, TracingConfig
from utils.logger import get_logger

from typing import Union, overload, Literal
from cremedelacreme import AsyncFlow, Flow
from nodes import (
    # Company laptop only nodes
    StatusQueryNode,
    # Online query nodes
    RedactInputNode,
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
    TicketCreationNode,
)

logger = get_logger(__name__)

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
        redact_node = RedactInputNode()
        intent_node = IntentClassificationNode()
        embed_node = EmbedQueryNode()
        search_node = SearchKnowledgeBaseNode()
        decision_node = DecisionMakerNode()
        answer_node = GenerateAnswerNode()
        clarify_node = AskClarifyingQuestionNode()
        format_node = FormatFinalResponseNode()
        create_ticket_node = TicketCreationNode()
        
        # Interactive troubleshooting node
        troubleshoot_node = InteractiveTroubleshootNode()
        
        # Placeholder nodes for not-yet-implemented features
        search_tickets_node = NotImplementedNode("Ticket search")
        
        # Connect nodes
        # Linear path: redact -> intent -> embed (with follow-up detection) -> search
        _ = redact_node >> intent_node >> embed_node >> search_node
        
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
        _ = search_tickets_node >> format_node
        _ = create_ticket_node >> format_node

        # Troubleshoot can exit, escalate, or continue (default)
        _ = troubleshoot_node >> format_node  # default: continue
        _ = troubleshoot_node - "exit" >> format_node
        _ = troubleshoot_node - "escalate" >> create_ticket_node
      
        
        # Initialize Flow with start node (redact first!)
        super().__init__(start=redact_node)
        
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
# Network Status Flow
# ============================================================================

@trace_flow(config=tracing_config, flow_name="NetworkStatusFlow")
class NetworkStatusFlow(AsyncFlow):
    """Checking network status flow."""
    def __init__(self):
        # Create nodes
        status_node = StatusQueryNode()
        
        # Connect in sequence
        _ = status_node
        
        # Initialize Flow with start node
        super().__init__(start=status_node)
        
        logger.info("Network status flow created with tracing")


def create_network_status_flow() -> AsyncFlow:
    """
    Create the network status flow.
    
    Returns:
        Configured AsyncFlow instance with tracing
    """
    return NetworkStatusFlow()


# ============================================================================
# Test Query Flow (for testing/development)
# ============================================================================

@trace_flow(config=tracing_config, flow_name="ITSupportTestQueryFlow")
class TestQueryFlow(Flow):
    """Test query flow for development - no agent routing, straight RAG pipeline."""
    def __init__(self):
        # Create nodes
        redact_node = RedactInputNode()
        intent_node = IntentClassificationNode()
        embed_node = EmbedQueryNode()
        query_node = StatusQueryNode()
        search_node = SearchKnowledgeBaseNode()
        answer_node = GenerateAnswerNode()
        format_node = FormatFinalResponseNode()
        
        # Simple linear flow
        _ = redact_node >> intent_node >> embed_node >> query_node >> search_node
        
        # Both search outcomes go to answer
        _ = search_node - "docs_found" >> answer_node
        _ = search_node - "no_docs" >> answer_node
        
        _ = answer_node >> format_node
        
        # Initialize Flow with start node
        super().__init__(start=redact_node)
        
        logger.info("Test query flow created with tracing")


def create_test_query_flow() -> Flow:
    """
    Test flow for RAG pipeline without agent routing.
    Goes straight from search to answer.
    
    Returns:
        Configured Flow instance with tracing
    """
    return TestQueryFlow()


# ============================================================================
# Flow Factory
# ============================================================================

_flow_cache: dict = {}


@overload
def get_flow(flow_type: Literal["status"]) -> AsyncFlow: ...

@overload
def get_flow(flow_type: Literal["query", "indexing", "test"]) -> Flow: ...

@overload
def get_flow(flow_type: str = "query") -> Union[Flow, AsyncFlow]: ...


def get_flow(flow_type: str = "query") -> Union[Flow, AsyncFlow]:
    """
    Get a flow by type name. Flows are cached for efficiency.
    
    Args:
        flow_type: One of "query", "indexing", "status", "test"
    
    Returns:
        Flow or AsyncFlow instance (cached)
        
    Raises:
        ValueError: If flow_type is unknown
    """
    if flow_type not in _flow_cache:
        flows = {
            "status": create_network_status_flow,
            "query": create_query_flow,
            "indexing": create_indexing_flow,
            "test": create_test_query_flow
        }
        
        if flow_type not in flows:
            raise ValueError(f"Unknown flow type: {flow_type}. Available: {list(flows.keys())}")
        
        _flow_cache[flow_type] = flows[flow_type]()
    
    return _flow_cache[flow_type]


if __name__ == "__main__":
    # Test flow creation
    print("Testing flow creation...")
    
    try:
        status_flow = create_network_status_flow()
        print(f"✓ Status flow created: {status_flow}")

        query_flow = create_query_flow()
        print(f"✓ Query flow created: {query_flow}")
        
        indexing_flow = create_indexing_flow()
        print(f"✓ Indexing flow created: {indexing_flow}")
        
        test_flow = create_test_query_flow()
        print(f"✓ Test query flow created: {test_flow}")
        
        print("\n✓ All flows created successfully")
        
    except Exception as e:
        print(f"\n✗ Error creating flows: {e}")
        import traceback
        traceback.print_exc()
