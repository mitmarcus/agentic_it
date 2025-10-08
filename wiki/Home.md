# Agentic IT Support Chatbot Wiki

Welcome to the **Agentic IT Support Chatbot** documentation! This wiki provides comprehensive information about the project, its architecture, and how to use and contribute to it.

## 🚀 Quick Links

- [Getting Started](Getting-Started.md) - Get up and running quickly
- [Installation](Installation.md) - Detailed installation instructions
- [Configuration](Configuration.md) - Configure the chatbot for your environment
- [Architecture](Architecture.md) - Understand the system design
- [API Documentation](API-Documentation.md) - API reference and usage
- [Troubleshooting](Troubleshooting.md) - Common issues and solutions
- [Contributing](Contributing.md) - How to contribute to the project

## 📖 What is Agentic IT?

The Agentic IT Support Chatbot is an intelligent assistant designed to help IT support teams handle common employee inquiries efficiently. It leverages:

- **RAG (Retrieval-Augmented Generation)** for answering questions from a knowledge base
- **Agentic workflows** for intelligent decision-making and routing
- **Interactive troubleshooting** for step-by-step problem resolution
- **Privacy-first design** with local embeddings and data redaction
- **Jira integration** for ticket management

## 🎯 Key Features

- **Quick Answers**: Retrieve information from IT documentation instantly
- **Guided Troubleshooting**: Interactive workflows for common IT issues
- **Major Incident Detection**: Check for ongoing widespread issues
- **Automated Ticket Creation**: Generate well-formatted support tickets
- **Privacy Protection**: Automatic redaction of sensitive information
- **Multi-turn Conversations**: Maintain context across the conversation

## 📊 Project Status

The project is actively being developed. Check the main [README](../README.md) for the latest updates.

**Latest Update**: 08oct25 - The agent is fully working!

## 🏗️ Architecture Overview

The chatbot uses the **cremedelacreme** framework for orchestrating nodes and flows:

```
User Query → Intent Classification → Decision Maker
              ↓                            ↓
       Search KB ← ─ ─ ─ ─ ─ ─ ┬ → Generate Answer
              ↓                 │
       Troubleshooting          │
              ↓                 │
       Search Tickets           │
              ↓                 │
       Create Ticket  → ─ ─ ─ ─ ┘
```

For detailed information, see the [Architecture](Architecture.md) page.

## 📚 Documentation

- **User Guides**: For end-users interacting with the chatbot
- **Developer Guides**: For developers working on the codebase
- **Deployment Guides**: For deploying and maintaining the system
- **API Reference**: For integrating with the chatbot programmatically

## 🤝 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](Troubleshooting.md) guide
2. Review the [design documentation](../docs/design.md)
3. Open an issue on GitHub

## 📝 License

See the main repository for license information.
