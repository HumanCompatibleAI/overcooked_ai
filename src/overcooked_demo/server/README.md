# Overcooked Demo Server

This directory contains the server code for the Overcooked AI demo.

## Running with uv

The server now uses [uv](https://github.com/astral-sh/uv) for dependency management and running Python code.

### Prerequisites

Install uv:

```bash
pip install uv
```

### Running the Server

You can run the server directly using:

```bash
uv run app.py
```

Alternatively, use the provided shell script:

```bash
./run_server.sh
```

## Docker

The Dockerfile has been updated to use uv as well. Build and run the Docker container with:

```bash
# Build the container
docker build -t overcooked-server \
  --build-arg BUILD_ENV=development \
  --build-arg OVERCOOKED_BRANCH=master \
  --build-arg GRAPHICS=js .

# Run the container
docker run -p 5000:5000 overcooked-server
```

## Dependencies

Dependencies are now managed through the `pyproject.toml` file. If you need to add or modify dependencies, update this file instead of the legacy `requirements.txt`. 