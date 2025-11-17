#!/usr/bin/env python3
"""
Test script for tracing functionality.

This script tests the tracing implementation to ensure it works correctly
with Langfuse integration.
"""

import sys
import os
import asyncio
import pytest #needed for @pytest.mark.asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from cremedelacreme import Node, Flow, AsyncNode, AsyncFlow
from langfuse_tracing import trace_flow, TracingConfig
from langfuse_tracing.utils.setup import setup_tracing

class SimpleNode(Node):
    """Simple test node for tracing verification."""

    def prep(self, shared):
        """Test prep phase."""
        return shared.get("input", "test_input")

    def exec(self, prep_res):
        """Test exec phase."""
        return f"processed_{prep_res}"

    def post(self, shared, prep_res, exec_res):
        """Test post phase."""
        shared["output"] = exec_res
        return "default"


class SimpleAsyncNode(AsyncNode):
    """Simple async test node for tracing verification."""

    async def prep_async(self, shared):
        """Test async prep phase."""
        await asyncio.sleep(0.1)  # Simulate async work
        return shared.get("input", "async_test_input")

    async def exec_async(self, prep_res):
        """Test async exec phase."""
        await asyncio.sleep(0.1)  # Simulate async work
        return f"async_processed_{prep_res}"

    async def post_async(self, shared, prep_res, exec_res):
        """Test async post phase."""
        shared["output"] = exec_res
        return "default"


@trace_flow(flow_name="TestSyncFlow")
class SyncFlowUnderTest(Flow):
    """Test synchronous flow with tracing."""

    def __init__(self):
        super().__init__(start=SimpleNode())


@trace_flow(flow_name="TestAsyncFlow")
class AsyncFlowUnderTest(AsyncFlow):
    """Test asynchronous flow with tracing."""

    def __init__(self):
        super().__init__(start=SimpleAsyncNode())


def test_sync_flow():
    """Test synchronous flow tracing."""
    print("🧪 Testing synchronous flow tracing...")

    flow = SyncFlowUnderTest()
    shared = {"input": "sync_test_data"}

    print(f"   Input: {shared}")
    result = flow.run(shared)
    print(f"   Output: {shared}")
    print(f"   Result: {result}")

    # Verify the flow worked
    assert "output" in shared
    assert shared["output"] == "processed_sync_test_data"
    print("   ✅ Sync flow test passed")


@pytest.mark.asyncio
async def test_async_flow():
    """Test asynchronous flow tracing."""
    print("🧪 Testing asynchronous flow tracing...")

    flow = AsyncFlowUnderTest()
    shared = {"input": "async_test_data"}

    print(f"   Input: {shared}")
    result = await flow.run_async(shared)
    print(f"   Output: {shared}")
    print(f"   Result: {result}")

    # Verify the flow worked
    assert "output" in shared
    assert shared["output"] == "async_processed_async_test_data"
    print("   ✅ Async flow test passed")


def test_configuration():
    """Test configuration loading and validation."""
    print("🧪 Testing configuration...")

    # Test loading from environment
    config = TracingConfig.from_env()
    print(f"   Loaded config: debug={config.debug}")

    # Test validation
    is_valid = config.validate()
    print(f"   Config valid: {is_valid}")

    if is_valid:
        print("   ✅ Configuration test passed")
    else:
        print(
            "   ⚠️ Configuration test failed (this may be expected if env vars not set)"
        )


def test_error_handling():
    """Test error handling in traced flows."""
    print("🧪 Testing error handling...")

    class ErrorNode(Node):
        def exec(self, prep_res):
            raise ValueError("Test error for tracing")

    @trace_flow(flow_name="TestErrorFlow")
    class ErrorFlow(Flow):
        def __init__(self):
            super().__init__(start=ErrorNode())

    flow = ErrorFlow()
    shared = {"input": "error_test"}

    try:
        flow.run(shared)
        print("   ❌ Expected error but flow succeeded")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        print(f"   ✅ Error correctly caught and traced: {e}")
        assert str(e) == "Test error for tracing"
    except Exception as e:
        print(f"   ⚠️ Unexpected error type: {e}")
        raise


# Standalone execution support
if __name__ == "__main__":
    async def main():
        """Run all tests."""
        # Test configuration first
        test_configuration()
        print()

        # Test setup (optional - only if environment is configured)
        try:
            print("Testing setup...")
            config = setup_tracing()
            print("✅ Setup test passed")
        except Exception as e:
            print(f"⚠️ Setup test failed: {e}")
            print("(This is expected if Langfuse is not configured)")
        print()

        # Test sync flow
        test_sync_flow()
        print()

        # Test async flow
        await test_async_flow()
        print()

        # Test error handling
        test_error_handling()
        print()

        print("🎉 All tests completed!")
        print("\n📊 If Langfuse is configured, check your dashboard for traces:")
        langfuse_host = os.getenv("LANGFUSE_HOST", "your-langfuse-host")
        print(f"   Dashboard URL: {langfuse_host}")

    asyncio.run(main())
