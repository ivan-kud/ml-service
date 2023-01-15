# Run JupyterLab
jupyter-lab

# Run TensorBoard
tensorboard --logdir=runs

# Run application locally
uvicorn src.app:app --reload

# Build docker image
docker build . -t ml-service

# Run docker container
docker run --rm -it -p 80:80 ml-service

# Login to Docker Hub
docker login -u ivankud

# Tag your image with your registry username
docker tag null/ml-service ivankud/ml-service

# Push image to Docker Hub repository
docker push ivankud/ml-service
