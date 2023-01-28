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



rm -r ml-service
git clone https://github.com/ivan-kud/ml-service.git
cd ml-service
docker network create traefik-public
docker compose -f compose.traefik.yaml up -d
docker compose -f compose.fastapi.yaml up -d



###############################################
# A f t e r   D r o p l e t   c r e a t i o n #
###############################################

### Run the service ###
# SSH to host as 'root'
ssh root@xxx.xxx.xxx.xxx
# Install updates (optional)
apt-get update && apt-get upgrade
# Clone repository
git clone https://github.com/ivan-kud/ml-service.git
# Go to directory
cd ml-service
# Build image and start container
docker compose up -d

### Install certificate ###
# Check that domain is working properly and stop service
docker compose down
# Add non-root user 'ml-service'
adduser ml-service
# Add your new user to the sudo group
usermod -aG sudo ml-service
# Examine the list of installed UFW profiles
ufw app list
# Allow SSH connections
ufw allow OpenSSH
# Enable the firewall
ufw enable
# You can see that SSH connections are still allowed
ufw status
# Copy your local public SSH key to new user to log in with SSH
rsync --archive --chown=ml-service:ml-service ~/.ssh /home/ml-service
# Now, open up a new terminal session, and use SSH to log in as new user
ssh ml-service@xxx.xxx.xxx.xxx
# Make sure your snapd core is up to date
sudo snap install core; sudo snap refresh core
# Remove certbot
sudo apt remove certbot
# Install certbot
sudo snap install --classic certbot
# link certbot command to your path, so you’ll be able to run it by just typing certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
# Open up the appropriate port(s) in your firewall
sudo ufw allow 80
sudo ufw allow 443
# Run Certbot to get our certificate
sudo certbot certonly --standalone -d ivankud.com
# When running the command, you will be prompted to enter an email address and agree to the terms of service.
# After doing so, you should see a message telling you the process was successful and where your certificates are stored
# Let’s take a look at what Certbot has downloaded for us
sudo ls /etc/letsencrypt/live/ivankud.com

### Handling Certbot Automatic Renewals ###
# Open the config file
sudo nano /etc/letsencrypt/renewal/ivankud.com.conf
# Add a hook on the last line that will reload any web-facing services, making them use the renewed certificate
"renew_hook = systemctl reload your_service"
# Save and close the file, then run a Certbot dry run to make sure the syntax is ok
sudo certbot renew --dry-run
# To see in detail, visit
# https://www.digitalocean.com/community/tutorials/how-to-use-certbot-standalone-mode-to-retrieve-let-s-encrypt-ssl-certificates-on-ubuntu-22-04
# https://certbot.eff.org/instructions

### Run the service again ###
# SSH to host as 'root'
ssh root@xxx.xxx.xxx.xxx
# Go to directory
cd ml-service
# Build image and start container
docker compose up -d

