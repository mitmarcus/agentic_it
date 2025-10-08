# Agentic IT Support Chatbot Wiki

This directory contains the comprehensive documentation for the Agentic IT Support Chatbot project.

## Wiki Structure

```
wiki/
├── Home.md                    # Main landing page with overview
├── Getting-Started.md         # Quick setup guide for new users
├── Installation.md            # Detailed installation instructions
├── Configuration.md           # Configuration options and environment variables
├── Architecture.md            # System architecture and design patterns
├── API-Documentation.md       # REST API reference and examples
├── Troubleshooting.md         # Common issues and solutions
└── Contributing.md            # Guidelines for contributors
```

## Documentation Pages

### [Home](Home.md)
Main entry point with project overview, quick links, and key features.

### [Getting Started](Getting-Started.md)
Quick start guide covering:
- Prerequisites
- Installation steps
- Basic configuration
- First query examples
- Verification steps

### [Installation](Installation.md)
Comprehensive installation guide including:
- System requirements
- Local development setup
- Docker installation
- Production deployment
- Post-installation steps

### [Configuration](Configuration.md)
Complete configuration reference:
- Environment variables
- LLM configuration (Groq)
- Embedding model settings
- ChromaDB configuration
- Document ingestion settings
- API configuration
- Advanced settings

### [Architecture](Architecture.md)
Technical architecture documentation:
- High-level architecture
- Core components
- Data flow diagrams
- Node system design
- Flow orchestration
- Storage architecture
- Design patterns

### [API Documentation](API-Documentation.md)
API reference and usage:
- Endpoints (chat, indexing, sessions)
- Request/response models
- Authentication
- Error handling
- Rate limiting
- Code examples (Python, JavaScript)

### [Troubleshooting](Troubleshooting.md)
Common issues and solutions:
- Installation issues
- Configuration problems
- Runtime errors
- Performance issues
- Database problems
- Debugging tips

### [Contributing](Contributing.md)
Contributor guidelines:
- Development setup
- Coding standards
- Testing guidelines
- Documentation standards
- Pull request process
- Review process

## Using the Wiki

### For End Users

Start with:
1. [Getting Started](Getting-Started.md) - Set up the chatbot
2. [API Documentation](API-Documentation.md) - Use the API
3. [Troubleshooting](Troubleshooting.md) - Fix common issues

### For Developers

Start with:
1. [Architecture](Architecture.md) - Understand the system
2. [Contributing](Contributing.md) - Development workflow
3. [Configuration](Configuration.md) - Advanced settings

### For Operators

Start with:
1. [Installation](Installation.md) - Deploy the system
2. [Configuration](Configuration.md) - Configure for production
3. [Troubleshooting](Troubleshooting.md) - Maintain the system

## Documentation Standards

When contributing to the wiki:

1. **Clarity**: Write clear, concise documentation
2. **Examples**: Include code examples and screenshots
3. **Structure**: Use consistent formatting and structure
4. **Links**: Link to related pages and sections
5. **Updates**: Keep documentation in sync with code

## Wiki vs Code Documentation

- **Wiki (here)**: User guides, tutorials, how-tos, architecture
- **Code docs**: Technical API docs, inline comments, docstrings
- **README**: Project overview, quick start, links
- **Design doc**: Detailed technical design (`/docs/design.md`)

## Viewing the Wiki

### In Repository

View markdown files directly in the `wiki/` directory.

### On GitHub

These files can be migrated to GitHub's wiki feature:
1. Go to repository Settings
2. Enable Wiki
3. Import these markdown files

### As Website

Convert to static site with tools like:
- [MkDocs](https://www.mkdocs.org/)
- [Docusaurus](https://docusaurus.io/)
- [GitBook](https://www.gitbook.com/)

## Contributing to Documentation

To improve the wiki:

1. Edit markdown files in `wiki/` directory
2. Follow the [Contributing](Contributing.md) guide
3. Submit a pull request
4. Maintainers will review and merge

## Feedback

Found an issue or have a suggestion? Please:
- Open an issue on GitHub
- Label it with `documentation`
- Provide specific details

## License

This documentation is part of the Agentic IT project and follows the same license.

---

**Start here**: [Home](Home.md) | [Getting Started](Getting-Started.md)
