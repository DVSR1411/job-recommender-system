@echo off
echo Starting Ollama container...
docker desktop start
docker run -d --name ollama -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama:0.9.3-rc1

echo Waiting for Ollama to start...
timeout /t 10 /nobreak

echo Pulling models from your Docker Hub...
docker exec ollama ollama pull nomic-embed-text:v1.5
docker exec ollama ollama pull qwen3:0.6b

echo Running ollama models in docker Hub...
docker exec ollama ollama run nomic-embed-text:v1.5
docker exec ollama ollama run qwen3:0.6b

echo Models ready!
echo Ollama API: http://localhost:11434