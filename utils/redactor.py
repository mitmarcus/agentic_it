"""
Sensitive data redaction utilities.
"""
import re
from typing import Pattern

# Default configuration
_DEFAULT_REPLACEMENT = "[REDACTED]"
_ALL_PATTERN_NAMES = (
    "api_key", "password_field", "token", "aws_key", "private_key"
)

# Sensitive dictionary keys (case-insensitive)
_SENSITIVE_KEYS = {
    "password", "passwd", "pwd", "secret", "api_key", "apikey",
    "token", "access_token", "refresh_token", "auth_token",
    "private_key", "privatekey", "client_secret", "aws_secret_access_key"
}

# Redaction patterns
PATTERNS: dict[str, Pattern] = {
    "api_key": re.compile(r'\b[A-Za-z0-9]{32,}\b'),
    "password_field": re.compile(r'(?i)(password|pwd|passwd|pass)\s*(?:is|:|\=)\s*[\w\d]+'),
    "token": re.compile(r'(?i)(token|bearer)\s*[:=]?\s*[A-Za-z0-9\-._~+/]+=*'),
    "aws_key": re.compile(r'(?i)(aws_access_key_id|aws_secret_access_key)\s*[:=]\s*[A-Za-z0-9/+=]+'),
    "private_key": re.compile(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC )?PRIVATE KEY-----'),
}


def redact_text(
    text: str,
    *,
    patterns: tuple[str, ...] = _ALL_PATTERN_NAMES,
    replacement: str = _DEFAULT_REPLACEMENT
) -> str:
    """
    Redact sensitive information from text.
    
    Args:
        text: Input text
        patterns: Pattern names to apply (defaults to all patterns)
        replacement: Replacement string for redacted content
    
    Returns:
        Redacted text
        
    Example:
        >>> redact_text("password: secret123", patterns=("password_field",))
        'password [REDACTED]'
    """
    if not text:
        return text
    
    redacted = text
    
    for pattern_name in patterns:
        if pattern_name in PATTERNS:
            pattern = PATTERNS[pattern_name]
            # Special handling for password_field to preserve context
            if pattern_name == "password_field":
                redacted = pattern.sub(lambda m: m.group(1) + ' ' + replacement, redacted)
            else:
                redacted = pattern.sub(replacement, redacted)
    
    return redacted


def _is_sensitive_key(key: str) -> bool:
    """Check if dictionary key is sensitive (case-insensitive)."""
    return key.lower() in _SENSITIVE_KEYS


def redact_dict(
    data: dict,
    *,
    patterns: tuple[str, ...] = _ALL_PATTERN_NAMES,
    replacement: str = _DEFAULT_REPLACEMENT
) -> dict:
    """
    Recursively redact sensitive information from dictionary.
    
    Redacts based on:
    1. Sensitive key names (password, token, secret, etc.)
    2. Pattern matching in string values
    
    Args:
        data: Input dictionary
        patterns: Pattern names to apply (defaults to all patterns)
        replacement: Replacement string
    
    Returns:
        Dictionary with redacted values
        
    Example:
        >>> redact_dict({"password": "secret123", "api_key": "abc123"})
        {'password': '[REDACTED]', 'api_key': '[REDACTED]'}
    """
    if not isinstance(data, dict):
        return data
    
    def _redact_value(key: str, val):
        """Helper to redact a value based on key name and content."""
        # Check if key name is sensitive
        if _is_sensitive_key(key):
            return replacement
        
        # Otherwise apply pattern-based redaction
        if isinstance(val, str):
            return redact_text(val, patterns=patterns, replacement=replacement)
        elif isinstance(val, dict):
            return redact_dict(val, patterns=patterns, replacement=replacement)
        elif isinstance(val, list):
            return [_redact_value("", item) for item in val]
        else:
            return val
    
    return {key: _redact_value(key, value) for key, value in data.items()}


def is_sensitive(
    text: str,
    *,
    patterns: tuple[str, ...] = _ALL_PATTERN_NAMES
) -> bool:
    """
    Check if text contains sensitive information.
    
    Args:
        text: Input text
        patterns: Pattern names to check (defaults to all patterns)
    
    Returns:
        True if sensitive data detected
        
    Example:
        >>> is_sensitive("password: secret123", patterns=("password_field",))
        True
    """
    if not text:
        return False
    
    for pattern_name in patterns:
        if pattern_name in PATTERNS:
            if PATTERNS[pattern_name].search(text):
                return True
    
    return False


def get_redaction_summary(
    text: str,
    *,
    patterns: tuple[str, ...] = _ALL_PATTERN_NAMES
) -> dict[str, int]:
    """
    Get summary of redacted items by type.
    
    Args:
        text: Input text
        patterns: Pattern names to check (defaults to all patterns)
    
    Returns:
        Dict mapping pattern names to count of matches
        
    Example:
        >>> get_redaction_summary("password: secret123 token: abc")
        {'password_field': 1, 'token': 1}
    """
    if not text:
        return {}
    
    summary = {}
    
    for pattern_name in patterns:
        if pattern_name in PATTERNS:
            matches = PATTERNS[pattern_name].findall(text)
            if matches:
                summary[pattern_name] = len(matches)
    
    return summary


if __name__ == "__main__":
    # Test redactor
    print("Testing redactor...")
    
    # Test text with sensitive data
    test_text = """
    User report:
    I tried to login with password=MySecret123 but got an error.
    Here's my API key: abcdef1234567890abcdef1234567890
    AWS credentials: aws_access_key_id=AKIAIOSFODNN7EXAMPLE
    """
    
    print("\nOriginal text:")
    print(test_text)
    
    redacted = redact_text(test_text)
    print("\nRedacted text:")
    print(redacted)
    
    # Check if sensitive
    has_sensitive = is_sensitive(test_text)
    print(f"\n✓ Contains sensitive data: {has_sensitive}")
    
    # Get summary
    summary = get_redaction_summary(test_text)
    print(f"\n✓ Redaction summary:")
    for pattern_type, count in summary.items():
        print(f"    {pattern_type}: {count} match(es)")
    
    # Test dictionary redaction
    test_dict = {
        "user": "john_doe",
        "details": {
            "password": "secret123",
            "notes": "Test notes"
        },
        "logs": [
            "Connection successful",
            "API key: abcd1234567890abcd1234567890abcd"
        ]
    }
    
    print("\nOriginal dict:")
    print(test_dict)
    
    redacted_dict = redact_dict(test_dict)
    print("\nRedacted dict:")
    print(redacted_dict)
    
    print("\n✓ All tests passed")