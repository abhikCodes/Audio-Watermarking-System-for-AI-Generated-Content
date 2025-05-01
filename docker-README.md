# Docker Setup for Audio Steganography Project

This README contains instructions for running the audio steganography project using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

## Project Structure

The project is containerized into two services:
- Backend (FastAPI Python service)
- Frontend (React application)

## Getting Started

1. Build and start the containers:

```bash
docker-compose up --build
```

2. Access the applications:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8001

## Configuration

The Docker setup includes:
- Persistent storage for uploads and processed files
- Network for container communication
- Environment configuration for API connectivity

## Troubleshooting

- If the frontend cannot connect to the backend, ensure the proxy settings in `package.json` or the environment variables point to the correct backend URL
- If model loading fails, check that the model paths are correctly mapped in the containers

## Stopping the Services

To stop the running containers:

```bash
docker-compose down
```

## Development

For development purposes:
- Backend code changes require rebuilding the container
- Frontend changes can be made directly in the mounted volumes

## Data Persistence

Uploaded and processed files are stored in volumes mapped to:
- `./backend/uploads`
- `./backend/processed` 