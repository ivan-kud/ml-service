###############################################
# A f t e r   D r o p l e t   c r e a t i o n #
###############################################
# Change DNS records to new IP
# SSH to host as 'root'
ssh root@ivankud.com
# or (if DNS records is still not updated)
ssh root@xxx.xxx.xxx.xxx
# Install updates (optional)
apt update && apt upgrade
# Clone repository
git clone https://github.com/ivan-kud/ml-service.git
# Add environment variables
export TRAEFIK_USERNAME=xxxxxxxx
export TRAEFIK_PASSWORD=xxxxxxxx
export TRAEFIK_HASHED_PASSWORD=$(openssl passwd -apr1 $PASSWORD)
# Create network
docker network create traefik-public
# Build images and start containers
cd ml-service
docker compose up -d

#################################################
# B e f o r e   D r o p l e t   d e l e t i o n #
#################################################
# SSH to host as 'root'
ssh root@ivankud.com
# Add environment variables
export USERNAME=xxxxxxxx
export PASSWORD=xxxxxxxx
export HASHED_PASSWORD=$(openssl passwd -apr1 $PASSWORD)
# Down Compose
cd ml-service
docker compose down
# Find certificates on remote server (optional)
find / -type f -iname "acme.json"
# Pull certificates to local machine (command should be executed on the local machine)
rsync -a root@ivankud.com:/var/lib/docker/volumes/ml-service_traefik-public-certificates/_data/ ~/Documents/GitHub/ml-service/certificates

#################################
# U s e f u l   C o m m a n d s #
#################################
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
rsync -a --chown=ml-service:ml-service ~/.ssh /home/ml-service
# Now, open up a new terminal session, and use SSH to log in as new user
ssh ml-service@ivankud.com

# Run JupyterLab
jupyter-lab

# Run TensorBoard
tensorboard --logdir=runs
