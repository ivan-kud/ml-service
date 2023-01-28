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

# Docker Compose
docker compose up -d
docker compose ps
docker compose logs
docker compose pause
docker compose unpause
docker compose stop
docker compose start
docker compose down

# Examine the list of installed UFW profiles
ufw app list
# Allow SSH connections and ports
ufw allow OpenSSH
ufw allow 80
ufw allow 443
# Enable the firewall
ufw enable
# See connections that are still allowed
ufw status

# Add non-root user 'ml-service'
adduser ml-service
# Add your new user to the sudo group
usermod -aG sudo ml-service
# Copy your local public SSH key to new user to log in with SSH
rsync --archive --chown=ml-service:ml-service ~/.ssh /home/ml-service
# Now, open up a new terminal session, and use SSH to log in as new user
ssh ml-service@xxx.xxx.xxx.xxx

###############################################
# A f t e r   D r o p l e t   c r e a t i o n #
###############################################
# SSH to host as 'root'
ssh root@ivankud.com
# Install updates (optional)
apt-get update && apt-get upgrade
# Clone repository
git clone https://github.com/ivan-kud/ml-service.git
# Go to directory
cd ml-service
# Add environment variables
export USERNAME=xxxxxx
export PASSWORD=xxxxxxxx
export HASHED_PASSWORD=$(openssl passwd -apr1 $PASSWORD)
# Create network
docker network create traefik-public
# Build images and start containers
docker compose up -d

#################################################
# B e f o r e   D r o p l e t   d e l e t i o n #
#################################################
docker compose down
rm -r ml-service
