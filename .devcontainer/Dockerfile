# Development Dockerfile for pdf_split_and_count with PostgreSQL
# For use with VS Code Remote Development

FROM python:3.13-slim AS base

# Prevent apt from prompting for input
ENV DEBIAN_FRONTEND=noninteractive

# Combined apt call to minimize layers and install essential packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    tree \
    vim \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create the developer user and set up permissions
RUN useradd -s /bin/bash -m developer && \
    usermod -aG sudo developer && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to developer user for remaining operations
USER developer

# Set up Python environment
ENV PATH=/home/developer/.local/bin:$PATH
WORKDIR /workspace

# Install Python packages
RUN python -m ensurepip --default-pip && \
    pip install --no-cache-dir --upgrade pip

# Install development tools
RUN pip install --no-cache-dir \
    build \
    GitPython \
    pdm \
    poppler-utils \
    pydantic \
    pytest \
    ruff \
    twine

# Switch back to root for final setup
USER root

# Switch back to developer user
USER developer
RUN git config --global --add safe.directory /mounted/dev/pdf_split_and_count