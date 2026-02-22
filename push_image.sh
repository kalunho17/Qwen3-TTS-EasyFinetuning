#!/bin/bash

# Default tag
TAG=${1:-latest}
IMAGE_NAME="ghcr.io/mozi1924/qwen3-tts-easyfinetuning"

echo "🚀 Starting local build and push for $IMAGE_NAME:$TAG"

# 1. Build the image locally
echo "📦 Building Docker image..."
docker build -t "$IMAGE_NAME:$TAG" .

if [ $? -ne 0 ]; then
    echo "❌ Build failed! Please check the errors above."
    exit 1
fi

# 2. Push to GHCR
echo "⬆️ Pushing to GitHub Container Registry..."
docker push "$IMAGE_NAME:$TAG"

if [ $? -ne 0 ]; then
    echo "❌ Push failed!"
    echo "💡 Make sure you are logged in to GHCR: 'docker login ghcr.io'"
    echo "💡 You might need to create a Personal Access Token (PAT) with 'write:packages' scope."
    exit 1
fi

echo "✅ Successfully pushed $IMAGE_NAME:$TAG"
