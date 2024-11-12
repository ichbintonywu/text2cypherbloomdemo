Here’s a copy-friendly version of the installation guide:

1. Local Environment Setup

Install Python 3.11 and Create Virtual Environment

# Upgrade pip
pip install --upgrade pip

# Create a virtual environment
python3.11 -m venv myenv

# Activate the virtual environment
# Linux/macOS
source myenv/bin/activate
# Windows
myenv\Scripts\activate

# Install necessary packages
pip install fsspec pypdf neo4j graphdatascience openai langchain_ollama langchain_core langchain_community langchain_openai redis langchain streamlit

2. Docker Setup for Redis Container

# Pull the Redis image
docker pull redis

# Run the Redis container
docker run --name redis-container -p 6379:6379 -d redis

3. AWS AMI Setup

Connect to EC2 Instance

	1.	Instance Information:
	•	Instance ID: i-0d884231d893caa1a (tonywudemo)
	•	Public DNS: ec2-3-107-95-58.ap-southeast-2.compute.amazonaws.com
	2.	Locate your private key file (tonywudemo.pem).
	3.	Change the permissions of the key file (if necessary):

chmod 400 "tonywudemo.pem"

	4.	Connect to the EC2 instance using SSH:

ssh -i "tonywudemo.pem" ec2-user@ec2-3-107-95-58.ap-southeast-2.compute.amazonaws.com

Install Dependencies on EC2

# Install Python 3.11
sudo dnf install python3.11

# Install Python pip
sudo yum -y install python-pip

Transfer Files to EC2

# Transfer the file to the EC2 instance
scp -i tonywudemo.pem /Users/tony/Documents/dev/neo4j/openai/text2cypherbloomdemo.zip ec2-user@ec2-3-107-95-58.ap-southeast-2.compute.amazonaws.com:/tmp

Set Up Python Environment on EC2

# SSH into EC2 (if not already connected)
ssh -i tonywudemo.pem ec2-user@ec2-3-107-95-58.ap-southeast-2.compute.amazonaws.com

# Activate the virtual environment
source myenv/bin/activate

# Install necessary packages
pip install fsspec pypdf neo4j graphdatascience openai langchain_ollama langchain_core langchain_community langchain_openai redis langchain streamlit

Summary

This guide covers setting up a local Python environment, configuring Redis with Docker, connecting to an AWS EC2 instance, and setting up the necessary dependencies in a virtual environment.

This version is ready for you to copy and paste as needed!
