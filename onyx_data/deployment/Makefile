# Onyx Deployment Makefile
# Usage: make [target]

COMPOSE_FILE := docker-compose.yml
COMPOSE_GPU_FILE := docker-compose.gpu.yml
COMPOSE := docker compose -f $(COMPOSE_FILE)
COMPOSE_GPU := docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_GPU_FILE)

.PHONY: run start stop restart logs logs-api logs-bg logs-web status pull upgrade clean clean-all reset shell-api shell-bg db help run-gpu start-gpu restart-gpu upgrade-gpu reset-gpu

# Default target - start existing containers
.DEFAULT_GOAL := start

# First time setup (pull images + create + start containers)
run:
	$(COMPOSE) up -d
	@echo "Onyx started. Access UI at: http://localhost:7000"

# Start existing containers
start:
	$(COMPOSE) start
	@echo "Onyx started. Access UI at: http://localhost:7000"

# Stop all containers
stop:
	$(COMPOSE) stop

# Restart all containers
restart:
	$(COMPOSE) restart
	@echo "Onyx restarted. Access UI at: http://localhost:3000"

# Tail logs for all services
logs:
	$(COMPOSE) logs -f

# Tail logs for specific services
logs-api:
	$(COMPOSE) logs -f api_server

logs-bg:
	$(COMPOSE) logs -f background

logs-web:
	$(COMPOSE) logs -f web_server

# Show status of all containers
status:
	$(COMPOSE) ps -a

# Pull latest images
pull:
	$(COMPOSE) pull

# Upgrade: stop, pull latest, and start
upgrade: stop pull run
	@echo "Upgrade complete"

# ============================================================================
# GPU Variants (for NVIDIA GPU systems)
# ============================================================================

# First time setup with GPU
run-gpu:
	$(COMPOSE_GPU) up -d
	@echo "Onyx started with GPU. Access UI at: http://localhost:7000"

# Start existing containers with GPU
start-gpu:
	$(COMPOSE_GPU) start
	@echo "Onyx started with GPU. Access UI at: http://localhost:7000"

# Restart with GPU
restart-gpu:
	$(COMPOSE_GPU) restart
	@echo "Onyx restarted with GPU. Access UI at: http://localhost:7000"

# Upgrade with GPU: stop, pull latest, and start
upgrade-gpu: stop pull run-gpu
	@echo "Upgrade complete (GPU)"

# Reset with GPU: stop, remove containers, delete volumes, and start fresh (DESTRUCTIVE)
reset-gpu:
	@echo "WARNING: This will delete ALL data including database!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	$(COMPOSE) down -v
	$(COMPOSE_GPU) up -d
	@echo "Onyx reset complete with GPU."

# Stop and remove containers (keeps volumes/data)
clean:
	$(COMPOSE) down
	@echo "Containers removed. Data volumes preserved. Run 'make run' to recreate."

# Stop, remove containers AND delete volumes (DESTRUCTIVE)
clean-all:
	@echo "WARNING: This will delete ALL data including database!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	$(COMPOSE) down -v
	@echo "All containers and data removed."

# Reset: stop, remove containers, delete volumes, and start fresh (DESTRUCTIVE)
reset:
	@echo "WARNING: This will delete ALL data including database!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	$(COMPOSE) down -v
	$(COMPOSE) up -d
	@echo "Onyx reset complete."

# Open shell in containers
shell-api:
	$(COMPOSE) exec api_server bash

shell-bg:
	$(COMPOSE) exec background bash

shell-web:
	$(COMPOSE) exec web_server sh

# Open psql shell to database
db:
	$(COMPOSE) exec relational_db psql -U postgres

# Help
help:
	@echo "Onyx Deployment Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  start      Start existing containers (default)"
	@echo "  run        First time setup (pull + create + start)"
	@echo "  stop       Stop all containers"
	@echo "  restart    Restart all containers"
	@echo "  logs       Tail logs for all services"
	@echo "  logs-api   Tail logs for api_server"
	@echo "  logs-bg    Tail logs for background service"
	@echo "  logs-web   Tail logs for web_server"
	@echo "  status     Show container status"
	@echo "  pull       Pull latest images"
	@echo "  upgrade    Stop, pull latest, and start"
	@echo "  clean      Stop and remove containers (keeps data)"
	@echo "  clean-all  Stop and DELETE ALL DATA (destructive)"
	@echo "  reset      Delete all data and restart fresh (destructive)"
	@echo "  shell-api  Open shell in api_server"
	@echo "  shell-bg   Open shell in background"
	@echo "  shell-web  Open shell in web_server"
	@echo "  db         Open psql shell to database"
	@echo "  help       Show this help"
	@echo ""
	@echo "GPU Targets (for NVIDIA GPU systems):"
	@echo "  run-gpu      First time setup with GPU acceleration"
	@echo "  start-gpu    Start existing containers with GPU"
	@echo "  restart-gpu  Restart with GPU"
	@echo "  upgrade-gpu  Stop, pull latest, and start with GPU"
	@echo "  reset-gpu    Delete all data and restart fresh with GPU"
