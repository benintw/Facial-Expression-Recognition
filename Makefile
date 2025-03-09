# Default config file paths
TRAIN_CONFIG ?= configs/training.yaml
VAL_CONFIG ?= configs/validation.yaml
PRED_CONFIG ?= configs/inference.yaml


# Download dataset
download_dataset:
	@echo "Downloading dataset..."
	@./scripts/download_dataset.sh
	@echo "Dataset downloaded successfully."

# Visualize dataset
visualize_dataset:
	@echo "Visualizing dataset..."
	@poetry run python scripts/visualize_dataset.py
	@echo "Dataset visualized successfully."

# Training command
train:
	@echo "Starting training..."
	@./scripts/train.sh 
	@echo "Training completed. Check logs/train_*.log for details."

# Validation command
validate:
	@echo "Starting validation..."
	@./scripts/validate.sh
	@echo "Validation completed. Check logs/validate_*.log for details."

# Prediction command
predict:
	@echo "Starting prediction..."
	@./scripts/predict.sh
	@echo "Prediction completed. Check predictions/ for outputs."

# Run all steps in sequence
all: download_dataset train validate predict
	@echo "All steps completed successfully."

# Clean generated files
clean:
	rm -rf outputs/*
	rm -rf __pycache__/*
	rm -rf data/*
	rm -rf checkpoints/*
	rm -rf logs/*
	rm -rf predictions/*

# Docker commands
build:
	@echo "Building Docker image..."
	@./docker-build.sh
	@echo "Docker image built successfully."

run:
	@echo "Running Docker container..."
	@./docker-run.sh
	@echo "Container execution completed."

stop:
	@echo "Stopping Docker containers..."
	@./docker-stop.sh
	@echo "Containers stopped."

check:
	poetry check

.PHONY: train validate predict all clean check download_dataset visualize_dataset build run stop
