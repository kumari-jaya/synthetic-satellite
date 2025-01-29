#!/bin/bash

# Script to build and push all Vortx Docker images

set -e

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Build and push Vortx Docker images"
    echo ""
    echo "Options:"
    echo "  -t, --tag TAG     Image tag (default: latest)"
    echo "  -p, --push        Push images after building"
    echo "  -h, --help        Display this help message"
    exit 1
}

# Default values
TAG="latest"
PUSH=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -t|--tag)
            TAG="$2"
            shift
            shift
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Function to build and optionally push an image
build_and_push() {
    local service=$1
    local dockerfile=$2
    local tag=$3
    
    echo "Building $service image..."
    docker build \
        -t "vortx/vortx-$service:$tag" \
        -f "$dockerfile" \
        --build-arg VERSION=$tag \
        .
    
    if [ "$PUSH" = true ]; then
        echo "Pushing $service image..."
        docker push "vortx/vortx-$service:$tag"
    fi
}

# Build images
echo "Starting build process..."

# API service
build_and_push "api" "docker/api/Dockerfile" "$TAG"

# ML service
build_and_push "ml" "docker/ml/Dockerfile" "$TAG"

# Worker service
build_and_push "worker" "docker/worker/Dockerfile" "$TAG"

# Documentation
build_and_push "docs" "docs/Dockerfile" "$TAG"

echo "Build process completed successfully"

if [ "$PUSH" = true ]; then
    echo "All images have been pushed to Docker Hub"
fi

# Print summary
echo ""
echo "Built images:"
echo "- vortx/vortx-api:$TAG"
echo "- vortx/vortx-ml:$TAG"
echo "- vortx/vortx-worker:$TAG"
echo "- vortx/vortx-docs:$TAG"

if [ "$PUSH" = false ]; then
    echo ""
    echo "To push these images, run:"
    echo "$0 -t $TAG -p"
fi 