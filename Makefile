# LLM Projects Makefile
# Manages both Parakeet (transcription) and Onyx services
#
# GPU Assignment:
#   - GTX 1080 Ti (device 0): Parakeet transcription
#   - GTX 1060 3GB (device 1): Onyx embeddings

TRANSCRIPTION_DIR := transcription
ONYX_DIR := onyx_data/deployment

.PHONY: run start stop restart status logs help \
        run-transcription start-transcription stop-transcription restart-transcription logs-transcription \
        run-onyx start-onyx stop-onyx restart-onyx logs-onyx \
        gpu

# Default target
.DEFAULT_GOAL := status

# ============================================
# Combined Commands
# ============================================

# First time setup - build and create all containers
run: run-transcription run-onyx
	@echo ""
	@echo "All services started!"
	@echo "  Transcription: http://localhost:7001"
	@echo "  Onyx:          http://localhost:7000"

# Start existing containers
start: start-transcription start-onyx
	@echo ""
	@echo "All services started!"
	@echo "  Transcription: http://localhost:7001"
	@echo "  Onyx:          http://localhost:7000"

# Stop all services
stop: stop-transcription stop-onyx
	@echo "All services stopped"

# Restart all services
restart: restart-transcription restart-onyx
	@echo ""
	@echo "All services restarted!"
	@echo "  Transcription: http://localhost:7001"
	@echo "  Onyx:          http://localhost:7000"

# Show status of all services
status:
	@echo "=== Transcription (Parakeet) ==="
	@$(MAKE) -C $(TRANSCRIPTION_DIR) status --no-print-directory
	@echo ""
	@echo "=== Onyx ==="
	@$(MAKE) -C $(ONYX_DIR) status --no-print-directory

# Tail logs for all services (runs in foreground)
logs:
	@echo "Use 'make logs-transcription' or 'make logs-onyx' to tail specific logs"

# ============================================
# Transcription (Parakeet) Commands
# ============================================

run-transcription:
	@echo "Building and starting Transcription service..."
	@$(MAKE) -C $(TRANSCRIPTION_DIR) run --no-print-directory

start-transcription:
	@echo "Starting Transcription service..."
	@$(MAKE) -C $(TRANSCRIPTION_DIR) start --no-print-directory

stop-transcription:
	@$(MAKE) -C $(TRANSCRIPTION_DIR) stop --no-print-directory

restart-transcription:
	@$(MAKE) -C $(TRANSCRIPTION_DIR) restart --no-print-directory

logs-transcription:
	@$(MAKE) -C $(TRANSCRIPTION_DIR) logs --no-print-directory

# ============================================
# Onyx Commands
# ============================================

run-onyx:
	@echo "Starting Onyx services (first time setup)..."
	@$(MAKE) -C $(ONYX_DIR) run --no-print-directory

start-onyx:
	@echo "Starting Onyx services..."
	@$(MAKE) -C $(ONYX_DIR) start --no-print-directory

stop-onyx:
	@$(MAKE) -C $(ONYX_DIR) stop --no-print-directory

restart-onyx:
	@$(MAKE) -C $(ONYX_DIR) restart --no-print-directory

logs-onyx:
	@$(MAKE) -C $(ONYX_DIR) logs --no-print-directory

# ============================================
# Utility Commands
# ============================================

# Show GPU usage
gpu:
	@nvidia-smi

# Help
help:
	@echo "LLM Projects Makefile"
	@echo ""
	@echo "Combined Commands:"
	@echo "  make status     Show status of all services (default)"
	@echo "  make run        First time setup (build + create + start all)"
	@echo "  make start      Start all existing containers"
	@echo "  make stop       Stop all services"
	@echo "  make restart    Restart all services"
	@echo ""
	@echo "Transcription (Parakeet) Commands:"
	@echo "  make run-transcription      First time setup"
	@echo "  make start-transcription    Start existing container"
	@echo "  make stop-transcription     Stop container"
	@echo "  make restart-transcription  Restart container"
	@echo "  make logs-transcription     Tail logs"
	@echo ""
	@echo "Onyx Commands:"
	@echo "  make run-onyx      First time setup"
	@echo "  make start-onyx    Start existing containers"
	@echo "  make stop-onyx     Stop containers"
	@echo "  make restart-onyx  Restart containers"
	@echo "  make logs-onyx     Tail logs"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make gpu        Show GPU usage (nvidia-smi)"
	@echo "  make help       Show this help"
	@echo ""
	@echo "URLs:"
	@echo "  Transcription: http://localhost:7001"
	@echo "  Onyx:          http://localhost:7000"
