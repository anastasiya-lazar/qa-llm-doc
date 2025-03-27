# Document QA System

An AI-powered document question-answering system that uses advanced language models, vector databases, and autonomous agents to provide intelligent responses to questions about uploaded documents.

## Features

- Document upload and processing (PDF, TXT)
- Vector similarity search using FAISS
- Question answering using OpenAI's GPT-4
- Complex query handling with specialized AI agents
- RESTful API with FastAPI
- Persistent storage for documents and metadata
- Docker support for easy deployment
- Makefile for common development tasks

## Architecture

The system is built with the following components:

1. **Document Processor**: Handles document uploads, text extraction, and chunking
2. **Vector Store**: Manages document embeddings and similarity search using FAISS
3. **QA Engine**: Processes questions using LangChain and OpenAI
4. **Agent System**: Handles complex queries using CrewAI agents
5. **Storage**: Manages document persistence and metadata
6. **API Layer**: Provides RESTful endpoints using FastAPI

## Prerequisites

- Python 3.11.11 (for local development)
- OpenAI API key
- Docker and Docker Compose (for containerized deployment)
- Virtual environment (recommended for local development)
- Make (for using Makefile commands)

## Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd document-qa-system
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. Install dependencies and set up the development environment:
   ```bash
   make install
   ```

### Docker Deployment

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd document-qa-system
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. Build and start the containers:
   ```bash
   make docker-all
   ```

## Usage

### Local Development

1. Start the development server:
   ```bash
   make dev
   ```

2. Access the API documentation at `http://localhost:8000/docs`

### Docker Deployment

1. The API will be available at `http://localhost:8000/docs`
2. Uploaded documents and storage will persist in the `uploads/` and `storage/` directories
3. To stop the containers:
   ```bash
   make docker-down
   ```

### Development Commands

The project includes a Makefile with common development commands:

```bash
# Show available commands
make help

# Development setup
make install      # Install dependencies
make dev         # Start development server
make dev-setup   # Complete development setup (install, format, lint, test)

# Docker commands
make docker-build  # Build Docker image
make docker-up    # Start Docker containers
make docker-down  # Stop Docker containers
make docker-logs  # View Docker container logs
make build-and-up # Build and start Docker containers in foreground

# Code quality
make format      # Format code with black and isort
make lint        # Run linters (black, isort, flake8, mypy)
make test        # Run tests
make clean       # Clean up temporary files

# Combined commands
make all         # Run all checks (install, lint, test, format)
make docker-all  # Build and start Docker containers
```

## API Endpoints

### Document Management

- `POST /documents/upload`: Upload and process a document
- `GET /documents`: List all uploaded documents

### Question Answering

- `POST /questions/ask`: Ask a question about uploaded documents
- `POST /agents/complex-query`: Process a complex query using AI agents

## Design Decisions

1. **Document Processing**:
   - Chunk-based approach for better context management
   - Support for PDF and TXT formats
   - Metadata tracking for better organization

2. **Vector Search**:
   - FAISS for efficient similarity search
   - OpenAI embeddings for high-quality vector representations
   - Configurable chunk size and overlap

3. **Agent System**:
   - Specialized agents for different tasks
   - Sequential processing for better control
   - Action tracking for transparency

4. **Storage**:
   - File-based storage for simplicity
   - JSON format for easy inspection
   - Separate storage for metadata and chunks

5. **Containerization**:
   - Docker support for easy deployment
   - Volume mounts for persistent storage
   - Health checks for container monitoring
   - Network isolation for security

6. **Development Workflow**:
   - Makefile for common tasks
   - Automated code formatting and linting
   - Comprehensive testing setup
   - Clean development environment management

## Future Improvements

1. **Performance**:
   - Implement caching for frequently accessed documents
   - Add batch processing for large document sets
   - Optimize vector search with better indexing

2. **Features**:
   - Add support for more document formats
   - Implement document versioning
   - Add user authentication and authorization
   - Support for document collections and tags

3. **Architecture**:
   - Move to a proper database for better scalability
   - Implement distributed processing for large documents
   - Add monitoring and logging
   - Add Kubernetes support for production deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.