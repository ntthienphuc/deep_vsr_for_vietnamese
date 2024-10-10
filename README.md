
# Lip Reading Pipeline with Curriculum Learning

## Overview

This project implements a lip reading pipeline using a video-only model enhanced with Curriculum Learning. The training progresses through multiple iterations with increasing word counts to steadily improve the model's performance. The entire pipeline is containerized using Docker to ensure consistency and ease of deployment across different environments. GPU support is included to accelerate the training process.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [1. Install Docker and NVIDIA Container Toolkit](#1-install-docker-and-nvidia-container-toolkit)
  - [2. Prepare Data and Models](#2-prepare-data-and-models)
  - [3. Build the Docker Image](#3-build-the-docker-image)
  - [4. Run the Pipeline Using Docker Compose](#4-run-the-pipeline-using-docker-compose)
- [Configuration](#configuration)
- [Running Docker Without Docker Compose](#running-docker-without-docker-compose)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Curriculum Learning:** Automates training over predefined word counts to enhance learning progressively.
- **Dynamic Batch Size Adjustment:** Reduces batch size by half upon encountering Out Of Memory (OOM) errors during training.
- **Early Stopping:** Terminates training when validation Word Error Rate (WER) flattens and the learning rate reaches its minimum value.
- **GPU Acceleration:** Utilizes NVIDIA GPUs to significantly speed up the training process.
- **Dockerized Environment:** Ensures consistent and reproducible setups across different systems.
- **Comprehensive Logging:** Provides detailed logs and checkpoints for monitoring and debugging.

## Prerequisites

Before setting up and running the pipeline, ensure you have the following installed on your system:

- **Docker:** [Install Docker](https://docs.docker.com/get-docker/)
- **NVIDIA Drivers:** Latest drivers installed on your host machine.
- **NVIDIA Container Toolkit:** [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Git:** (Optional) For cloning the repository.

## Setup Instructions

Follow these steps to set up and run the lip reading pipeline using Docker and Docker Compose with GPU support.

### 1. Install Docker and NVIDIA Container Toolkit

#### **Install Docker**

- **Windows & macOS:** Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop).
- **Linux:** Follow the official [Docker Engine installation guide](https://docs.docker.com/engine/install/).

#### **Install NVIDIA Container Toolkit**

To enable GPU support within Docker containers, install the NVIDIA Container Toolkit.

1. **Prerequisites:**
   - NVIDIA GPU with the latest drivers installed.
   - Docker Engine installed.

2. **Installation Steps:**

   Follow the [official NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for your operating system.

3. **Verify Installation:**

   Run the following command to verify that Docker can access the GPU:

   ```bash
   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

   You should see the GPU details if the setup is correct.

### 2. Prepare Data and Models

**Data Preparation:**

- Place your CSV annotation files in the appropriate directory.
- Place your MP4 video files in the appropriate directory.
- Ensure that each video has a corresponding CSV file with the same base name.

**Model Files:**

- If you have pre-trained models, place them in the models/ directory.
- Ensure that all necessary model files (.pt files) are correctly named and located.

### 3. Build the Docker Image

From the root directory of your project, build the Docker image using the provided Dockerfile.

```bash
docker build -t lipreading_pipeline_gpu .
```

**Explanation:**

- `-t lipreading_pipeline_gpu`: Tags the image with the name lipreading_pipeline_gpu.
- `.`: Specifies the current directory as the build context.

### 4. Run the Pipeline Using Docker Compose

The provided docker-compose.yml simplifies running the Docker container with the necessary configurations.

**Steps:**

- Ensure Docker Compose is installed.
- Configure docker-compose.yml (if necessary).

To start the Docker container, run:

```bash
docker-compose up
```

Or, to run in detached mode:

```bash
docker-compose up -d
```

### Configuration

All configurations are managed through the config.py file. Adjust the parameters as needed, such as curriculum word counts, batch size, and learning rate.

### Running Docker Without Docker Compose

If you prefer not to use Docker Compose, you can run the Docker container directly with the necessary configurations.

```bash
docker run --gpus all -it --rm     -v $(pwd)/data:/app/data     -v $(pwd)/models:/app/models     lipreading_pipeline_gpu
```

## Troubleshooting

- **Docker Build Fails:** Ensure dependencies are correctly listed and system requirements are met.
- **GPU Not Detected:** Check NVIDIA driver and Container Toolkit installation.
- **Out of Memory (OOM) Errors:** Reduce batch size or free up GPU resources.
- **Permission Denied Errors:** Check file permissions and avoid running the container as root.

## Contributing

Contributions are welcome! If you'd like to enhance the pipeline or fix bugs, please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Create a pull request.
