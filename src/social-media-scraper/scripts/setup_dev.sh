#!/bin/bash

# Development environment setup script
set -e

echo "🛠️  Setting up Social Media Scraper Development Environment"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $PYTHON_VERSION == *"Python 3."* ]]; then
    MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d' ' -f2 | cut -d'.' -f1)
    MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d' ' -f2 | cut -d'.' -f2)
    
    if [[ $MAJOR_VERSION -eq 3 && $MINOR_VERSION -ge 8 ]]; then
        print_success "Python version: $PYTHON_VERSION ✓"
    else
        print_error "Python 3.8+ required, found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found"
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
print_status "Installing development dependencies..."
pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy pre-commit

# Install package in development mode
print_status "Installing package in development mode..."
pip install -e .

# Create necessary directories
print_status "Creating project directories..."
mkdir -p output logs tests/fixtures docs/images

# Copy environment template
if [ ! -f .env ]; then
    cp .env.example .env
    print_warning "Created .env file from template - please configure your API keys"
else
    print_warning ".env file already exists"
fi

# Set up pre-commit hooks
print_status "Setting up pre-commit hooks..."
if [ -f .pre-commit-config.yaml ]; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
fi

# Run initial tests
print_status "Running initial tests..."
pytest tests/ -v --tb=short || print_warning "Some tests failed - this is expected if API keys are not configured"

# Check code formatting
print_status "Checking code formatting..."
black --check src/ tests/ || print_warning "Code formatting issues found - run 'black src/ tests/' to fix"

print_success "Development environment setup complete!"
echo ""
echo "🚀 Quick Start Commands:"
echo "======================="
echo "  Activate environment: source venv/bin/activate"
echo "  Run development server: uvicorn src.social_scraper.main:app --reload"
echo "  Run tests: pytest"
echo "  Format code: black src/ tests/"
echo "  Lint code: flake8 src/ tests/"
echo "  Type check: mypy src/"
echo ""
echo "📝 Next Steps:"
echo "============="
echo "1. Configure API keys in .env file"
echo "2. Run the development server"
echo "3. Visit http://localhost:8000/docs for API documentation"
echo "4. Start developing new features!"

# =====================================
# Makefile
# =====================================
.PHONY: help install test lint format type-check clean dev prod docker-build docker-run

# Default target
help:
	@echo "Social Media Scraper - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies and setup development environment"
	@echo "  dev         Run development server"
	@echo "  test        Run tests"
	@echo "  test-cov    Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint        Run linting (flake8)"
	@echo "  format      Format code (black, isort)"
	@echo "  type-check  Run type checking (mypy)"
	@echo "  quality     Run all quality checks"
	@echo ""
	@echo "Production:"
	@echo "  prod        Run production server"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run  Run Docker container"
	@echo "  deploy      Deploy to production"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       Clean up temporary files"
	@echo "  docs        Generate documentation"

# Development setup
install:
	@echo "Setting up development environment..."
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install -e .[dev]
	@echo "Setup complete! Activate with: source venv/bin/activate"

# Development server
dev:
	uvicorn src.social_scraper.main:app --reload --port 8000

# Production server
prod:
	uvicorn src.social_scraper.main:app --host 0.0.0.0 --port 8000

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/ --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

quality: lint type-check
	@echo "All quality checks passed!"

# Docker
docker-build:
	docker build -t social-scraper:latest .

docker-run:
	docker run -d --name social-scraper --env-file .env -p 8000:8000 social-scraper:latest

# Production deployment
deploy:
	@./scripts/deploy.sh

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# Documentation
docs:
	@echo "API documentation available at http://localhost:8000/docs when server is running"