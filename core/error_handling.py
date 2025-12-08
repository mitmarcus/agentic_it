"""
Standardized error handling decorator for API endpoints.

Provides consistent error handling, logging, and HTTP exception formatting
across all FastAPI endpoints.
"""
from functools import wraps
from typing import Callable, TypeVar, Union, Any
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request as StarletteRequest
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


# ============================================================================
# Custom Exception Handlers
# ============================================================================

async def validation_exception_handler(
    request: Union[Request, StarletteRequest, Any], 
    exc: Union[RequestValidationError, Exception]
) -> JSONResponse:
    """
    Custom handler for Pydantic validation errors.
    
    Prevents exposure of internal implementation details (field names,
    validation types, constraints) that could help attackers map the
    attack surface.
    
    Instead of exposing:
    {
      "detail": [{
        "type": "string_too_long",
        "loc": ["body", "query"],
        "msg": "String should have at most 10000 characters",
        "ctx": {"max_length": 10000}
      }]
    }
    
    Returns user-friendly error:
    {
      "error": "Message too long",
      "message": "Your message is too long. Please shorten it and try again."
    }
    
    Security Benefits:
    - No field structure exposure (["body", "query"])
    - No validation type exposure (string_too_long, string_too_short)
    - No constraint values (max_length, min_length)
    - No internal parameter names
    - User-friendly plain English messages
    
    Args:
        request: The incoming request
        exc: The validation error exception
        
    Returns:
        JSONResponse with generic error message
    """
    # Log the actual error for debugging (internal use only)
    logger.warning(
        f"Validation error on {request.url.path}: {exc.errors()}",
        extra={"url": str(request.url), "errors": exc.errors()}
    )
    
    # Check error types and return appropriate user-friendly messages
    errors = exc.errors()
    for error in errors:
        error_type = error.get("type", "")
        
        # Handle empty/too-short input
        if error_type == "string_too_short":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Empty message",
                    "message": "Please enter a question or describe your IT issue."
                }
            )
        
        # Handle too-long input
        elif error_type == "string_too_long":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Message too long",
                    "message": "Your message is too long. Please shorten it and try again."
                }
            )
        
        # Handle missing required fields
        elif error_type == "missing":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Missing information",
                    "message": "Required information is missing. Please check your request."
                }
            )
        
        # Handle value errors (e.g., from custom validators)
        elif error_type == "value_error":
            # Use the custom error message from the validator
            msg = error.get("msg", "Invalid input value")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Invalid input",
                    "message": msg
                }
            )
    
    # Generic error for other validation issues
    # (Don't expose what went wrong specifically)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid request",
            "message": "Please check your input and try again."
        }
    )


async def json_decode_exception_handler(
    request: Union[Request, StarletteRequest, Any], 
    exc: Exception
) -> JSONResponse:
    """
    Handle JSON parsing errors without exposing parser details.
    
    Prevents exposure of:
    - JSON parser type/version
    - Exact syntax error location
    - Internal parsing logic
    
    Args:
        request: The incoming request
        exc: The JSON decode exception
        
    Returns:
        JSONResponse with generic error message
    """
    logger.warning(
        f"JSON decode error on {request.url.path}: {exc}",
        extra={"url": str(request.url)}
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid request format",
            "message": "Unable to process your request. Please check the format and try again."
        }
    )
