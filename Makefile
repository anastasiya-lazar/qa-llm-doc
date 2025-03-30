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
	docker-compose down -v
	docker-compose rm -f
	# Remove only project-specific images
	docker images "qa_openai-*" -q | xargs docker rmi -f || true

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