.PHONY: help install dev docker-build docker-up docker-down docker-logs build-and-up clean test lint format

# Default target
help:
	@echo "Document QA System Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies in virtual environment"
	@echo "  make dev        - Start development server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up   - Start Docker containers"
	@echo "  make docker-down - Stop Docker containers"
	@echo "  make docker-logs - View Docker container logs"
	@echo "  make build-and-up - Build and start Docker containers in foreground"
	@echo "  make clean      - Clean up temporary files"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo ""

# Development setup
install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

dev:
	. venv/bin/activate && uvicorn src.channel.fastapi.main:app --reload

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

build-and-up:
	docker-compose up --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name ".venv" -exec rm -rf {} +
	find . -type d -name "venv" -exec rm -rf {} +
	find . -type d -name ".env" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	find . -type f -name ".DS_Store" -delete

# Testing and code quality
test:
	. venv/bin/activate && pytest

lint:
	. venv/bin/activate && \
	black . --check && \
	isort . --check-only && \
	flake8 . && \
	mypy .

format:
	. venv/bin/activate && \
	black . && \
	isort .

# Combined commands
all: install lint test format

docker-all: docker-build docker-up

# Development workflow
dev-setup: install format lint test 