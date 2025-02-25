# Migrating to uv Package Manager

This project has been configured to use [uv](https://github.com/astral-sh/uv), a fast Python package manager. The following guide explains how to set up and use uv with this project.

## Installing uv

To install uv, run:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This adds uv to your system. You may need to restart your shell or add it to your PATH.

## Setup for Development

To set up a development environment:

```sh
# Clone the repository
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project in development mode with all dependencies
uv pip install -e ".[harl]"
```

## Common uv Commands

Replace pip commands with uv equivalents:

| pip | uv |
|-----|-----|
| `pip install package` | `uv pip install package` |
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| `pip freeze > requirements.txt` | `uv pip freeze > requirements.txt` |

## Benefits of uv

- Faster installation speed
- Better dependency resolution
- Improved caching
- Single binary with no dependencies

## Project Structure

The project now uses a `pyproject.toml` file for configuration, which is compatible with uv and modern Python packaging standards. The `setup.py` file has been maintained for backward compatibility.

## CI/CD Integration

GitHub Actions workflows have been updated to use uv for dependency installation. 