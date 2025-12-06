"""
Standardized error handling decorator for API endpoints.

Provides consistent error handling, logging, and HTTP exception formatting
across all FastAPI endpoints.
"""
from functools import wraps
from typing import Callable, TypeVar
from fastapi import HTTPException
from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

def handle_api_errors(operation_name: str):
    """
    Decorator to standardize error handling across API endpoints.
    
    This decorator:
    - Catches and logs all exceptions
    - Converts ValueError to 400 (bad request)
    - Converts general exceptions to 500 (internal server error)
    - Re-raises HTTPException as-is (already formatted)
    - Adds operation context to logs
    - Handles both sync and async functions
    
    Args:
        operation_name: Name of the operation (e.g., "query", "upload")
    
    Usage:
        @app.post("/query")
        @handle_api_errors("query")
        async def query_endpoint(...):
            if invalid_input:
                raise ValueError("Input validation failed")
            # ... process request
    
    Example:
        # Before (with manual error handling):
        @app.post("/query")
        async def query(request: QueryRequest):
            try:
                # ... logic
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
        
        # After (with decorator):
        @app.post("/query")
        @handle_api_errors("query")
        async def query(request: QueryRequest):
            # ... logic (no try/except needed)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        import asyncio
        import inspect
        
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except HTTPException:
                    # Re-raise HTTP exceptions (already formatted)
                    # These include our custom validation errors
                    raise
                except ValueError as e:
                    # Client errors (bad input validation)
                    logger.warning(
                        f"Validation error in {operation_name}: {e}",
                        extra={"operation": operation_name, "error_type": "validation"}
                    )
                    raise HTTPException(status_code=400, detail=str(e))
                except Exception as e:
                    # Server errors (unexpected failures)
                    logger.error(
                        f"Error in {operation_name}: {e}",
                        exc_info=True,
                        extra={"operation": operation_name, "error_type": "server"}
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing {operation_name}: {str(e)}"
                    )
            return async_wrapper  # type: ignore
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except HTTPException:
                    # Re-raise HTTP exceptions (already formatted)
                    # These include our custom validation errors
                    raise
                except ValueError as e:
                    # Client errors (bad input validation)
                    logger.warning(
                        f"Validation error in {operation_name}: {e}",
                        extra={"operation": operation_name, "error_type": "validation"}
                    )
                    raise HTTPException(status_code=400, detail=str(e))
                except Exception as e:
                    # Server errors (unexpected failures)
                    logger.error(
                        f"Error in {operation_name}: {e}",
                        exc_info=True,
                        extra={"operation": operation_name, "error_type": "server"}
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing {operation_name}: {str(e)}"
                    )
            return sync_wrapper  # type: ignore
    return decorator
