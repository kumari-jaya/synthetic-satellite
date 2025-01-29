#!/bin/bash

# Script to migrate Docker images from TileFormer to Vortx

set -e

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Migrate Docker images from TileFormer to Vortx"
    echo ""
    echo "Options:"
    echo "  -t, --tag TAG     Specific tag to migrate (default: all tags)"
    echo "  -h, --help        Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -t|--tag)
            TAG="$2"
            shift
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

# Function to migrate a single tag
migrate_tag() {
    local tag=$1
    echo "Migrating tag: $tag"
    
    # Pull old image
    docker pull "vortx/tileformer:$tag" || {
        echo "Error: Failed to pull vortx/tileformer:$tag"
        return 1
    }
    
    # Tag with new name
    docker tag "vortx/tileformer:$tag" "vortx/vortx:$tag" || {
        echo "Error: Failed to tag vortx/tileformer:$tag as vortx/vortx:$tag"
        return 1
    }
    
    # Push new image
    docker push "vortx/vortx:$tag" || {
        echo "Error: Failed to push vortx/vortx:$tag"
        return 1
    }
    
    echo "Successfully migrated tag: $tag"
}

# Main migration logic
echo "Starting Docker image migration..."

if [ -n "$TAG" ]; then
    # Migrate specific tag
    migrate_tag "$TAG"
else
    # Get all tags
    tags=$(docker images "vortx/tileformer" --format "{{.Tag}}")
    
    if [ -z "$tags" ]; then
        echo "No TileFormer images found"
        exit 0
    fi
    
    # Migrate each tag
    for tag in $tags; do
        migrate_tag "$tag"
    done
fi

echo "Migration completed successfully"

# Cleanup instructions
echo ""
echo "To clean up old images, run:"
echo "docker images | grep 'vortx/tileformer' | awk '{print \$3}' | xargs docker rmi" 