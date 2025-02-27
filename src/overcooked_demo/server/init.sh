#!/bin/bash

# Create the agents directory if it doesn't exist
mkdir -p /app/data/agents

# Start the application
exec "$@" 