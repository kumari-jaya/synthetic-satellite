#!/bin/bash

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)/.."
export CUDA_VISIBLE_DEVICES=""  # Set to empty for CPU-only testing

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install test requirements
pip install -r requirements-test.txt

# Parse command line arguments
PYTEST_ARGS=""
COVERAGE=false
BENCHMARK=false
INTEGRATION=false
GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            COVERAGE=true
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --integration)
            INTEGRATION=true
            shift
            ;;
        --gpu)
            GPU=true
            export CUDA_VISIBLE_DEVICES="0"
            shift
            ;;
        --runslow)
            PYTEST_ARGS="$PYTEST_ARGS --runslow"
            shift
            ;;
        *)
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
    esac
done

# Build pytest command
CMD="pytest"

if [ "$COVERAGE" = true ]; then
    CMD="$CMD --cov=tileformer --cov-report=html --cov-report=term"
fi

if [ "$BENCHMARK" = true ]; then
    CMD="$CMD --benchmark-only --benchmark-autosave"
fi

if [ "$INTEGRATION" = true ]; then
    CMD="$CMD --integration"
fi

if [ "$GPU" = true ]; then
    CMD="$CMD --gpu"
fi

# Add common options
CMD="$CMD -v --tb=short -n auto --maxfail=2 --durations=10"

# Add any remaining arguments
CMD="$CMD $PYTEST_ARGS"

# Print test configuration
echo "Running tests with configuration:"
echo "  Coverage: $COVERAGE"
echo "  Benchmark: $BENCHMARK"
echo "  Integration: $INTEGRATION"
echo "  GPU: $GPU"
echo "  Additional args: $PYTEST_ARGS"
echo "  Command: $CMD"
echo

# Run tests
eval $CMD

# Generate coverage report if enabled
if [ "$COVERAGE" = true ]; then
    echo
    echo "Generating coverage report..."
    coverage html
    echo "Coverage report generated in htmlcov/index.html"
fi

# Deactivate virtual environment
deactivate 