# Run JupyterLab
jupyter-lab

# Run TensorBoard
tensorboard --logdir=runs

# Run application locally
cd app
uvicorn src.app:app --reload

# Build docker image
docker build . -t ivankud/ml-service

# Run docker container
docker run --rm -it -p 80:80 ivankud/ml-service

# Login to Docker Hub
docker login -u ivankud

# If needed, tag your image with your registry username
docker tag null/ml-service ivankud/ml-service

# Push image to Docker Hub repository
docker push ivankud/ml-service

# Pull image form Docker Hub repository
docker pull ivankud/ml-service:latest

# Run Docker Compose in background
docker compose up -d

docker compose ps
docker compose logs
docker compose pause
docker compose unpause
docker compose stop
docker compose start

# Remove the containers, networks, and volumes
docker compose down
