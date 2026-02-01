@echo off
echo Starting Ollama container...
docker desktop start
docker run -d --name ollama -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama:0.9.3-rc1

echo Pulling models from your Docker Hub..
docker exec ollama ollama pull nomic-embed-text:v1.5
docker exec ollama ollama pull llama3.2:3b

echo Running models from your Docker Hub...
docker exec ollama ollama run llama3.2:3b
docker exec ollama ollama run nomic-embed-text:v1.5

echo Models ready!
echo Ollama API: http://localhost:11434