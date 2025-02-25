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

# Clone chai code for the human_aware_rl module
RUN apt-get update && apt-get install -y git
RUN git clone --recursive https://github.com/HumanCompatibleAI/overcooked_ai.git --branch $OVERCOOKED_BRANCH --single-branch /overcooked_ai

# Dummy data_dir so things don't break
RUN echo "import os; DATA_DIR=os.path.abspath('.')" >> /overcooked_ai/src/human_aware_rl/data_dir.py

# Install server dependencies
RUN . .venv/bin/activate && uv pip install -e .

# Install the cloned overcooked_ai in development mode
RUN . .venv/bin/activate && uv pip install -e /overcooked_ai

# Install eventlet production server if production build
RUN if [ "$BUILD_ENV" = "production" ] ; then . .venv/bin/activate && uv pip install "eventlet>=0.39.0" ; fi

# Install needed packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
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
CMD [".venv/bin/python", "-m", "dev_helper", "app.py"]
