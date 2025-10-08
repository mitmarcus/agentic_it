# Contributing Guide

Thank you for your interest in contributing to the Agentic IT Support Chatbot! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing opinions and experiences
- Accept responsibility and apologize for mistakes

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.9+
- Git
- A GitHub account
- Familiarity with Python and FastAPI

### Areas to Contribute

We welcome contributions in various areas:

1. **New Features**
   - Additional node types
   - New integrations (Jira, Slack, etc.)
   - Enhanced RAG capabilities
   - UI improvements

2. **Bug Fixes**
   - Fix reported issues
   - Improve error handling
   - Performance optimizations

3. **Documentation**
   - Improve guides and tutorials
   - Add code examples
   - Fix typos and clarifications

4. **Testing**
   - Write unit tests
   - Add integration tests
   - Improve test coverage

5. **Infrastructure**
   - CI/CD improvements
   - Docker optimizations
   - Deployment scripts

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/agentic_it.git
cd agentic_it

# Add upstream remote
git remote add upstream https://github.com/mitmarcus/agentic_it.git
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies including dev dependencies
pip install -r requirements.txt

# If there's a requirements-dev.txt
pip install -r requirements-dev.txt
```

### 4. Set Up Environment

```bash
cp .env.sample .env
# Edit .env with your configuration
```

### 5. Verify Setup

```bash
# Run tests
python -m pytest tests/ -v

# Test flow creation
python flows.py

# Start server
python main.py
```

## How to Contribute

### Finding Issues to Work On

1. Check the [Issues](https://github.com/mitmarcus/agentic_it/issues) page
2. Look for labels:
   - `good first issue` - Great for beginners
   - `help wanted` - Community help needed
   - `bug` - Bug fixes needed
   - `enhancement` - New features

3. Comment on the issue to let others know you're working on it

### Creating a New Issue

Before creating an issue, check if it already exists. Include:

- **Bug Reports:**
  - Clear description of the bug
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (OS, Python version, etc.)
  - Error messages and logs

- **Feature Requests:**
  - Clear description of the feature
  - Use case and benefits
  - Potential implementation approach
  - Any relevant examples

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

```python
# Good
def search_knowledge_base(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search the knowledge base for relevant documents.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of document dictionaries with scores
    """
    # Implementation
    pass

# Use type hints
from typing import List, Dict, Optional

# Use descriptive names
user_query = "How do I reset my password?"
retrieved_docs = search_kb(user_query)

# Constants in UPPERCASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
```

### Code Organization

```python
# Order of imports
import os                    # Standard library
import logging               # Standard library

from typing import List      # Standard library typing

import chromadb              # Third-party
from fastapi import FastAPI  # Third-party

from nodes import MyNode     # Local modules
from utils import helper     # Local utils
```

### Documentation

All functions and classes should have docstrings:

```python
def process_document(doc: Dict, chunk_size: int = 1000) -> List[Dict]:
    """
    Process a document into chunks for indexing.
    
    This function splits a document into smaller chunks while preserving
    context and metadata. It handles various document formats.
    
    Args:
        doc: Document dictionary with 'content' and 'metadata'
        chunk_size: Maximum characters per chunk (default: 1000)
        
    Returns:
        List of chunk dictionaries, each with:
            - content: Chunk text
            - metadata: Original metadata plus chunk_index
            
    Raises:
        ValueError: If document format is invalid
        
    Example:
        >>> doc = {'content': 'VPN guide...', 'metadata': {...}}
        >>> chunks = process_document(doc, chunk_size=500)
        >>> len(chunks)
        5
    """
    pass
```

### Node Development Guidelines

When creating new nodes:

```python
from cremedelacreme import Node
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MyNewNode(Node):
    """
    Brief description of what this node does.
    
    This node is used in [flow_name] to accomplish [purpose].
    """
    
    def __init__(self, param1: str = "default"):
        """Initialize node with configuration."""
        super().__init__()
        self.param1 = param1
        logger.info(f"MyNewNode initialized with param1={param1}")
    
    def prep(self, shared: Dict) -> Any:
        """
        Read from shared store.
        
        Reads:
            - shared["input_key"]
            
        Returns:
            Prepared data for exec()
        """
        logger.debug(f"Preparing MyNewNode")
        return shared.get("input_key")
    
    def exec(self, prep_res: Any) -> Any:
        """
        Core processing logic (stateless).
        
        Args:
            prep_res: Result from prep()
            
        Returns:
            Processed result
        """
        logger.info(f"Executing MyNewNode")
        # Core logic here
        result = self._process(prep_res)
        return result
    
    def post(self, shared: Dict, prep_res: Any, exec_res: Any) -> str:
        """
        Write to shared store and return edge.
        
        Writes:
            - shared["output_key"]
            
        Returns:
            Edge name ("default", "success", "failure", etc.)
        """
        shared["output_key"] = exec_res
        logger.debug(f"MyNewNode completed")
        return "default"
    
    def _process(self, data: Any) -> Any:
        """Private helper method."""
        # Implementation
        pass
```

## Testing Guidelines

### Writing Tests

Tests should be clear, focused, and independent:

```python
import pytest
from nodes import MyNewNode

def test_my_new_node_success():
    """Test successful execution of MyNewNode."""
    # Arrange
    node = MyNewNode(param1="test")
    shared = {"input_key": "test data"}
    
    # Act
    prep_res = node.prep(shared)
    exec_res = node.exec(prep_res)
    edge = node.post(shared, prep_res, exec_res)
    
    # Assert
    assert shared["output_key"] == "expected result"
    assert edge == "default"

def test_my_new_node_failure():
    """Test MyNewNode with invalid input."""
    node = MyNewNode()
    shared = {"input_key": None}
    
    with pytest.raises(ValueError):
        prep_res = node.prep(shared)
        node.exec(prep_res)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_nodes.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test
python -m pytest tests/test_nodes.py::test_my_new_node_success -v
```

### Test Coverage

Aim for:
- **Unit tests**: > 80% coverage
- **Integration tests**: Critical paths
- **End-to-end tests**: Main user flows

## Documentation

### Types of Documentation

1. **Code Comments**
   - Explain "why", not "what"
   - Clarify complex logic
   - Document assumptions

2. **Docstrings**
   - All public functions/classes
   - Follow Google or NumPy style
   - Include examples

3. **README Updates**
   - Update if behavior changes
   - Add new features to list

4. **Wiki Pages**
   - Create guides for new features
   - Update existing pages
   - Add troubleshooting tips

### Documentation Style

- Use clear, concise language
- Provide code examples
- Include error handling examples
- Keep formatting consistent

## Submitting Changes

### Branch Naming

Use descriptive branch names:

```bash
# Feature
git checkout -b feature/add-slack-integration

# Bug fix
git checkout -b fix/memory-leak-in-embeddings

# Documentation
git checkout -b docs/update-installation-guide

# Refactor
git checkout -b refactor/simplify-node-logic
```

### Commit Messages

Write clear commit messages:

```bash
# Good
git commit -m "Add Slack notification support for ticket creation"
git commit -m "Fix memory leak in embedding cache"
git commit -m "Update installation guide with Docker instructions"

# Bad
git commit -m "fix bug"
git commit -m "update"
git commit -m "wip"
```

Format:
```
Brief summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"

Resolves: #123
```

### Pull Request Process

1. **Update your fork:**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Push to your fork:**
```bash
git push origin feature/your-feature-name
```

3. **Create Pull Request:**
   - Go to GitHub and create PR
   - Fill out the PR template
   - Link related issues
   - Describe changes clearly

4. **PR Template:**
```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings

## Related Issues
Closes #123
```

## Review Process

### What to Expect

1. **Automated Checks:**
   - Tests must pass
   - Linting must pass
   - Coverage requirements met

2. **Code Review:**
   - Maintainers will review
   - Address feedback promptly
   - Be open to suggestions

3. **Approval:**
   - At least one approval needed
   - All comments resolved
   - CI passing

### Responding to Feedback

- Thank reviewers for their time
- Ask for clarification if needed
- Make requested changes
- Explain reasoning if you disagree
- Update the PR when ready

### After Merge

- Delete your branch
- Update your fork
- Celebrate! 🎉

## Development Tips

### Local Testing

```bash
# Test specific component
python -c "from nodes import MyNode; node = MyNode(); print(node)"

# Interactive testing
python
>>> from flows import create_query_flow
>>> flow = create_query_flow()
>>> result = flow.run({'user_query': 'test', 'session_id': 'debug'})
```

### Debugging

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Use Python debugger
import pdb; pdb.set_trace()

# Or use breakpoint() (Python 3.7+)
breakpoint()
```

### Performance Testing

```python
import time

def benchmark_function():
    start = time.time()
    # Your code here
    duration = time.time() - start
    print(f"Execution time: {duration:.2f}s")
```

## Questions?

If you have questions:

1. Check existing documentation
2. Search closed issues
3. Ask in discussions
4. Open a new issue

Thank you for contributing to Agentic IT! 🚀
