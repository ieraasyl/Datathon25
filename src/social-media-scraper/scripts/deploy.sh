#!/bin/bash

# Production deployment script for Social Media Scraper
set -e

echo "🚀 Starting Social Media Scraper Deployment"
echo "============================================="

# Configuration
PROJECT_NAME="social-media-scraper"
DOCKER_IMAGE="social-scraper:latest"
CONTAINER_NAME="social-scraper-prod"
PORT=${PORT:-8000}

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

# Check if .env file exists
if [ ! -f .env ]; then
    print_error ".env file not found!"
    print_warning "Please create .env file with your API keys:"
    cat .env.example
    exit 1
fi

print_status "Environment file found ✓"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running!"
    exit 1
fi

print_status "Docker is running ✓"

# Build Docker image
print_status "Building Docker image..."
docker build -t $DOCKER_IMAGE .

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Stop existing container if running
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    print_warning "Stopping existing container..."
    docker stop $CONTAINER_NAME
fi

# Remove existing container if exists
if docker ps -aq -f name=$CONTAINER_NAME | grep -q .; then
    print_warning "Removing existing container..."
    docker rm $CONTAINER_NAME
fi

# Create output and logs directories
mkdir -p output logs

# Run new container
print_status "Starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    --env-file .env \
    -p $PORT:8000 \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    $DOCKER_IMAGE

if [ $? -eq 0 ]; then
    print_success "Container started successfully"
else
    print_error "Failed to start container"
    exit 1
fi

# Wait for container to be ready
print_status "Waiting for service to be ready..."
sleep 10

# Health check
HEALTH_URL="http://localhost:$PORT/health"
if curl -f -s $HEALTH_URL > /dev/null; then
    print_success "Service is healthy and ready!"
    print_status "API Documentation: http://localhost:$PORT/docs"
    print_status "Health Check: $HEALTH_URL"
else
    print_error "Service health check failed"
    print_status "Checking container logs..."
    docker logs $CONTAINER_NAME --tail 50
    exit 1
fi

# Show deployment summary
echo ""
echo "🎉 Deployment Complete!"
echo "======================="
echo "Service URL: http://localhost:$PORT"
echo "API Docs: http://localhost:$PORT/docs"
echo "Container: $CONTAINER_NAME"
echo "Image: $DOCKER_IMAGE"
echo ""
echo "Useful commands:"
echo "  View logs: docker logs $CONTAINER_NAME -f"
echo "  Stop service: docker stop $CONTAINER_NAME"
echo "  Restart service: docker restart $CONTAINER_NAME"
echo "  Remove service: docker rm -f $CONTAINER_NAME"

# =====================================
# scripts/setup_dev.sh
# =====================================
