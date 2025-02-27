#!/bin/sh

# Check for local flag
if [[ $1 = local* ]]; then
    echo "Running locally with uv"
    
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo "uv could not be found. Please install it first using 'pip install uv'"
        exit 1
    fi
    
    # Change to the server directory 
    cd "$(dirname "$0")/server"
    
    echo "Installing dependencies..."
    # Create a venv if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv
    fi
    
    # Activate the virtual environment
    source .venv/bin/activate
    
    # Install dependencies from pyproject.toml
    echo "Installing dependencies from pyproject.toml..."
    uv pip install -e .
    
    # Also install root package if it's not already installed
    echo "Ensuring overcooked_ai is installed..."
    ROOT_DIR="$(realpath "$(dirname "$0")/../..")"
    uv pip install -e "$ROOT_DIR"
    
    # Try to install tensorflow with a fallback
    echo "Attempting to install tensorflow..."
    if ! uv pip install -e ".[tensorflow]" 2>/dev/null; then
        echo "Could not install latest tensorflow. Trying compatible version..."
        # Try a version known to work on many platforms
        if ! uv pip install "tensorflow>=2.8.0" 2>/dev/null; then
            echo "Warning: Could not install tensorflow. The server will run with limited functionality."
            echo "You may need to manually install a compatible version of tensorflow."
        fi
    fi
    
    echo "Starting server..."
    # Use the Python interpreter from the venv to run the app with dev_helper
    .venv/bin/python -m dev_helper app.py
    
    exit 0
fi

mkdir -p server/overcooked_ai
cp -r ../../* server/overcooked_ai/
# Docker mode
if [[ $1 = prod* ]]; then
    echo "production"
    export BUILD_ENV=production

    # Completely re-build all images from scratch without using build cache
    # Copy the overcooked_ai directory from the project root to the server directory
    docker compose build --no-cache
    docker compose up --force-recreate -d
else
    echo "development"
    export BUILD_ENV=development
    # Uncomment the following line if there has been an update to overcooked-ai code
    # docker compose build --no-cache

    # Force re-build of all images but allow use of build cache if possible
    docker compose up --build
fi