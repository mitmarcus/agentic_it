"""
Setup and testing utilities for tracing.
"""

import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

from langfuse_tracing import TracingConfig, LangfuseTracer


def setup_tracing(env_file: Optional[str] = None) -> TracingConfig:
    """
    Set up tracing configuration and validate the setup.
    
    Args:
        env_file: Optional path to .env file. If None, uses default location.
        
    Returns:
        TracingConfig instance.
        
    Raises:
        RuntimeError: If setup fails.
    """
    print("Setting up tracing...")
    
    # Check if langfuse is installed
    if not LANGFUSE_AVAILABLE:
        raise RuntimeError(
            "Langfuse package not installed. Install with: pip install langfuse"
        )
    
    # Load configuration
    if env_file:
        config = TracingConfig.from_env(env_file)
        print(f"✓ Loaded configuration from: {env_file}")
    else:
        config = TracingConfig.from_env()
        print("✓ Loaded configuration from environment")
    
    # Validate configuration
    if not config.validate():
        raise RuntimeError(
            "Invalid tracing configuration. Please check your environment variables:\n"
            "- LANGFUSE_SECRET_KEY\n"
            "- LANGFUSE_PUBLIC_KEY\n" 
            "- LANGFUSE_HOST"
        )
    
    print("✓ Configuration validated")
    
    # Test connection
    if test_langfuse_connection(config):
        print("✓ Langfuse connection successful")
    else:
        raise RuntimeError("Failed to connect to Langfuse. Check your configuration and network.")
    
    print("🎉 tracing setup complete!")
    return config


def test_langfuse_connection(config: TracingConfig) -> bool:
    """
    Test connection to Langfuse.
    
    Args:
        config: TracingConfig instance.
        
    Returns:
        True if connection successful, False otherwise.
    """
    try:
        # Create a test tracer
        tracer = LangfuseTracer(config)
        
        if not tracer.client:
            return False
        
        # Try to start and end a test trace
        trace_id = tracer.start_trace("test_connection", {"test": True})
        if trace_id:
            tracer.end_trace({"test": "completed"}, "success")
            tracer.flush()
            return True
        
        return False
        
    except Exception as e:
        if config.debug:
            print(f"Connection test failed: {e}")
        return False



if __name__ == "__main__":
    """Command-line interface for setup and testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Tracing Setup")
    parser.add_argument("--test", action="store_true", help="Test Langfuse connection")
    parser.add_argument("--env-file", type=str, help="Path to .env file")
    
    args = parser.parse_args()
    
    if args.test:
        try:
            config = setup_tracing(args.env_file)
            print("\n✅ All setup tests passed! Your tracing is ready.")
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            print("\nFor help with configuration, run:")
            print("python utils/setup.py --help-config")
            sys.exit(1)
