FROM python:3.10-slim-bookworm

# System deps
RUN apt update && apt install -y default-jre htop build-essential && apt clean

# Environment variables for Neo4j
ENV NEO4J_HOST="neo4j://feature-discovery-neo4j:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="password"
ENV NEO4J_DATABASE="benchmark"

# Set workdir
WORKDIR /app

# Copy whole project
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Editable install (requires correct pyproject.toml + src/)
RUN pip install -e .
