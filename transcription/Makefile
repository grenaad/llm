# Whisper ASR Transcription Service
# Usage: make [target]

IMAGE_NAME := whisper-webui
CONTAINER_NAME := whisper-web
PORT := 7001

.PHONY: build run start stop restart logs clean shell status gpu rebuild

# Default target - start existing container
.DEFAULT_GOAL := start

# Build the Docker image
build:
	docker build --network=host -t $(IMAGE_NAME) .

# Run a new container (first time setup)
run: build
	docker run -d --gpus all \
		-p $(PORT):7860 \
		--shm-size=8GB \
		--name $(CONTAINER_NAME) \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-v ~/.cache/whisper:/root/.cache/whisper \
		-v ~/.local/share/whisper-transcriptions:/app/data \
		$(IMAGE_NAME)
	@echo "Container started. Model loading (~2 min)..."
	@echo "Access UI at: http://localhost:$(PORT)"

# Start existing container
start:
	docker start $(CONTAINER_NAME)
	@echo "Container started. Access UI at: http://localhost:$(PORT)"

# Stop the container
stop:
	docker stop $(CONTAINER_NAME)

# Restart the container
restart:
	docker restart $(CONTAINER_NAME)
	@echo "Container restarting. Model will reload (~2 min)..."

# Tail container logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Stop and remove container
clean:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "Container removed. Run 'make run' to create a new one."

# Open shell inside container
shell:
	docker exec -it $(CONTAINER_NAME) bash

# Show container status
status:
	@docker ps -a --filter "name=$(CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Show GPU usage
gpu:
	nvidia-smi

# Force rebuild without cache
rebuild:
	docker build --no-cache --network=host -t $(IMAGE_NAME) .

# Help
help:
	@echo "Whisper ASR Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  start    Start existing container (default)"
	@echo "  run      Build image and create new container"
	@echo "  stop     Stop the container"
	@echo "  restart  Restart the container"
	@echo "  logs     Tail container logs"
	@echo "  status   Show container status"
	@echo "  shell    Open bash shell in container"
	@echo "  clean    Stop and remove container"
	@echo "  build    Build Docker image"
	@echo "  rebuild  Force rebuild without cache"
	@echo "  gpu      Show GPU usage (nvidia-smi)"
	@echo "  help     Show this help"
