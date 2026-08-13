# Use official Python lightweight image
FROM python:3.12-slim

# Copy uv binaries from the official uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy dependency definition files to take advantage of docker layer caching
COPY pyproject.toml /app/

# Install dependencies (without installing the project source itself)
RUN uv sync --no-install-project

# Copy the rest of the application code
COPY . /app

# Expose port 8090
EXPOSE 8090

# Run uvicorn inside the uv synced virtualenv
CMD ["/app/.venv/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
