#!/bin/bash

echo "==== Building and starting Docker containers ===="
docker-compose up -d --build

echo "==== Waiting for services to start (30 seconds) ===="
sleep 30

echo "==== Testing backend API ===="
backend_response=$(curl -s http://localhost:8001)
echo "Backend response: $backend_response"

if [[ $backend_response == *"Audio Steganography API is running"* ]]; then
    echo "✅ Backend API is running correctly"
else
    echo "❌ Backend API test failed"
fi

echo "==== Testing frontend availability ===="
frontend_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
echo "Frontend HTTP status: $frontend_response"

if [[ $frontend_response == "200" ]]; then
    echo "✅ Frontend is accessible"
else
    echo "❌ Frontend test failed"
fi

echo "==== Docker container status ===="
docker-compose ps

echo "==== Docker logs ===="
echo "=== Backend logs ==="
docker-compose logs --tail=20 backend
echo "=== Frontend logs ==="
docker-compose logs --tail=20 frontend

echo "==== To stop the containers, run: docker-compose down ====" 