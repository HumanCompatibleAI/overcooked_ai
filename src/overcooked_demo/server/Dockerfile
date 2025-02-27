FROM python:3.9-slim 

ARG BUILD_ENV
ARG OVERCOOKED_BRANCH
ARG GRAPHICS

WORKDIR /app

# Install uv
RUN pip install uv

# Create a virtual environment for uv
RUN uv venv

# Copy the pyproject.toml for installation
COPY ./pyproject.toml ./pyproject.toml
COPY ./dev_helper.py ./dev_helper.py

# Install necessary packages for build
RUN apt-get update && apt-get install -y git build-essential

# Copy local overcooked_ai directory instead of git cloning
COPY ./overcooked_ai /overcooked_ai

# Dummy data_dir so things don't break
RUN echo "import os; DATA_DIR=os.path.abspath('.')" >> /overcooked_ai/src/human_aware_rl/data_dir.py

# Install server dependencies
RUN . .venv/bin/activate && uv pip install -e .

# Install a specific version of Ray that's compatible with the overcooked_ai codebase
RUN . .venv/bin/activate && uv pip install "ray[rllib]>=2.5.0"

# Install the cloned overcooked_ai in development mode
RUN . .venv/bin/activate && uv pip install -e /overcooked_ai

# Install eventlet production server if production build
RUN if [ "$BUILD_ENV" = "production" ] ; then . .venv/bin/activate && uv pip install "eventlet>=0.39.0" ; fi

# Install needed packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy over remaining files
COPY ./static ./static
COPY ./*.py ./
COPY ./graphics/$GRAPHICS ./static/js/graphics.js
COPY ./config.json ./config.json

# Set environment variables that will be used by app.py
ENV HOST 0.0.0.0
ENV PORT 5000
ENV CONF_PATH config.json

# Expose port and run the app, using dev_helper to ensure proper Python path
EXPOSE 5000
# CMD [".venv/bin/python", "-m", "dev_helper", "app.py"]
CMD ["uv", "run", "app.py"]
