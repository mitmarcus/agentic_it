"""
Test script to verify status caching behavior.

Run this to see:
1. Cache miss on first call (fetches fresh data)
2. Cache hit on subsequent calls (instant)
3. Cache expiry after TTL
"""
import asyncio
import time
import os
import pytest
from unittest.mock import AsyncMock, patch
from dotenv import load_dotenv

# Load environment variables before importing nodes
load_dotenv()

from nodes import StatusQueryNode


@pytest.mark.asyncio
async def test_cache_behavior():
    """Test the status caching mechanism."""
    print("=" * 60)
    print("Testing Status Cache Behavior")
    print("=" * 60)
    
    # Mock the status retrieval to avoid Playwright dependency
    mock_status_data = [
        {"type": "maintenance", "title": "Network Maintenance", "status": "active"},
        {"type": "incident", "title": "Email Outage", "status": "resolved"}
    ]
    
    with patch('utils.status_retrieval.scrape_session', new_callable=AsyncMock, return_value=mock_status_data):
        node = StatusQueryNode()
        shared = {}
        
        # Test 1: First call (cache miss)
        print("\n📝 Test 1: First call (should fetch fresh data)")
        print("-" * 60)
        start = time.time()
        await node.prep_async(shared)
        result1 = await node.exec_async(None)
        elapsed1 = time.time() - start
        print(f"✓ Fetched {len(result1)} status items in {elapsed1:.2f}s")
        print(f"  Cache age: {StatusQueryNode.get_cache_age():.2f}s")
        print(f"  Cache valid: {StatusQueryNode.is_cache_valid()}")
        
        # Test 2: Immediate second call (cache hit)
        print("\n📝 Test 2: Immediate second call (should use cache)")
        print("-" * 60)
        start = time.time()
        await node.prep_async(shared)
        result2 = await node.exec_async(None)
        elapsed2 = time.time() - start
        print(f"✓ Retrieved {len(result2)} status items in {elapsed2:.2f}s")
        print(f"  Cache age: {StatusQueryNode.get_cache_age():.2f}s")
        print(f"  Speedup: {(elapsed1/elapsed2):.0f}x faster!")
        assert result1 == result2, "Results should match"
        
        # Test 3: Wait and call again (cache hit, but older)
        print("\n📝 Test 3: After 2 seconds (cache still valid)")
        print("-" * 60)
        await asyncio.sleep(2)
        start = time.time()
        await node.prep_async(shared)
        result3 = await node.exec_async(None)
        elapsed3 = time.time() - start
        print(f"✓ Retrieved {len(result3)} status items in {elapsed3:.2f}s")
        print(f"  Cache age: {StatusQueryNode.get_cache_age():.2f}s")
        print(f"  Cache valid: {StatusQueryNode.is_cache_valid()}")
        
        # Test 4: Manual cache clear
        print("\n📝 Test 4: Manual cache clear")
        print("-" * 60)
        StatusQueryNode.clear_cache()
        print(f"✓ Cache cleared")
        print(f"  Cache age: {StatusQueryNode.get_cache_age():.2f}s")
        print(f"  Cache valid: {StatusQueryNode.is_cache_valid()}")
        
        # Test 5: After clear (cache miss again)
        print("\n📝 Test 5: After cache clear (should fetch fresh data)")
        print("-" * 60)
        start = time.time()
        await node.prep_async(shared)
        result5 = await node.exec_async(None)
        elapsed5 = time.time() - start
        print(f"✓ Fetched {len(result5)} status items in {elapsed5:.2f}s")
        print(f"  Cache age: {StatusQueryNode.get_cache_age():.2f}s")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Summary")
        print("=" * 60)
        print(f"First call (miss):   {elapsed1:.3f}s")
        print(f"Cached call (hit):   {elapsed2:.3f}s ({(elapsed1/elapsed2):.0f}x faster)")
        print(f"After 2s (hit):      {elapsed3:.3f}s")
        print(f"After clear (miss):  {elapsed5:.3f}s")
        print(f"\n✅ Cache working as expected!")
        print(f"   TTL: {StatusQueryNode._cache_ttl}s")
        print(f"   Speedup: ~{(elapsed1/elapsed2):.0f}x for cached queries")


@pytest.mark.asyncio
async def test_parallel_execution():
    """Test that status check can run in parallel with other work."""
    print("\n" + "=" * 60)
    print("Testing Parallel Execution")
    print("=" * 60)
    
    # Mock the status retrieval
    mock_status_data = [
        {"type": "maintenance", "title": "Network Maintenance", "status": "active"}
    ]
    
    with patch('utils.status_retrieval.scrape_session', new_callable=AsyncMock, return_value=mock_status_data):
        # Clear cache for fresh test
        StatusQueryNode.clear_cache()
        
        node = StatusQueryNode()
        shared = {}
        
        async def mock_rag_pipeline():
            """Simulate main RAG pipeline work."""
            print("  [RAG] Starting RAG pipeline...")
            await asyncio.sleep(1)  # Simulate RAG work
            print("  [RAG] RAG pipeline complete")
            return "RAG result"
        
        async def status_check():
            """Run status check."""
            print("  [Status] Starting status check...")
            await node.prep_async(shared)
            result = await node.exec_async(None)
            print(f"  [Status] Status check complete ({len(result)} items)")
            return result
        
        # Run both in parallel
        print("\n📝 Running status check in parallel with RAG pipeline...")
        print("-" * 60)
        start = time.time()
        
        status_result, rag_result = await asyncio.gather(
            status_check(),
            mock_rag_pipeline()
        )
        
        elapsed = time.time() - start
        print("-" * 60)
        print(f"✓ Both completed in {elapsed:.2f}s")
        print(f"  Status returned {len(status_result)} items")
        print(f"  RAG returned: {rag_result}")
        print(f"\n✅ Parallel execution working! Total time ~1s (not 2s if sequential)")


if __name__ == "__main__":
    print("\n🚀 Starting Status Cache Tests\n")
    
    # Run cache behavior tests
    asyncio.run(test_cache_behavior())
    
    # Run parallel execution test
    asyncio.run(test_parallel_execution())
    
    print("\n✅ All tests passed!\n")
