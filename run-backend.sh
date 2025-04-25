#!/bin/bash

# Change to backend directory
cd backend

# Activate virtual environment if needed
# source venv/bin/activate 

# Run the server with the correct module path (using port 8001)
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload 