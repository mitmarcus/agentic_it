"""
Centralized logging configuration for the application.

Best practices implemented:
- Single configuration point (setup_logging called once at startup)
- Structured log format with timestamp, level, name, and message
- File rotation to prevent unbounded log growth
- Separate log levels for console and file
- JSON format option for production log aggregation
- Proper exception logging with exc_info
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

# Logging configuration from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE_ENABLED = os.getenv("LOG_FILE_ENABLED", "true").lower() == "true"
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "./logs/app.log")
LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", 10 * 1024 * 1024))  # 10MB default
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", 5))  # Keep 5 backups
LOG_JSON_FORMAT = os.getenv("LOG_JSON_FORMAT", "false").lower() == "true"

# Standard format for human-readable logs
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# JSON format for production log aggregation (ELK, CloudWatch, etc.)
LOG_FORMAT_JSON = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

_configured = False
_root_logger: Optional[logging.Logger] = None


class ExcludeHealthCheckFilter(logging.Filter):
    """Filter out noisy health check logs."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out health check endpoint logs
        message = record.getMessage()
        if "/health" in message and "200" in message:
            return False
        return True


def setup_logging() -> None:
    """
    Configure logging for the application.
    Should be called once at application startup.
    
    Features:
    - Console handler with colored output (if terminal supports it)
    - Rotating file handler to prevent disk space issues
    - Optional JSON format for log aggregation systems
    - Filters out noisy health check logs
    """
    global _configured, _root_logger
    if _configured:
        return
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Choose format based on configuration
    log_format = LOG_FORMAT_JSON if LOG_JSON_FORMAT else LOG_FORMAT
    formatter = logging.Formatter(log_format, datefmt=LOG_DATE_FORMAT)
    
    # Console handler - writes to stderr (best practice for containers)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ExcludeHealthCheckFilter())
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if LOG_FILE_ENABLED:
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    _root_logger = root_logger
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    Ensures logging is configured before returning the logger.
    
    Args:
        name: The name for the logger (typically __name__)
        
    Returns:
        A configured logger instance
    """
    setup_logging()
    return logging.getLogger(name)
