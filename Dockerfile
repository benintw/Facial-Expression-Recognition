# Use an official python runtime as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git for cloning dataset, other tools as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install poetry 
RUN pip install --no-cache-dir poetry

# copy only dependency files first
COPY . .

# install dependencies
RUN poetry install --no-interaction --no-ansi


# ensure scripts are executable
RUN chmod +x scripts/*.sh

EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "make", "all"]
