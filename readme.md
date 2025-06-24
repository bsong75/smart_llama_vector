## Create containers for Llama and Frontend creation with Gradio
docker-compose up -d

## Pull the model, using Ollama
### Any model you want to pull 
docker exec -it llama-container ollama run llama3.2:3b

docker-compose up -d --build --force-recreate