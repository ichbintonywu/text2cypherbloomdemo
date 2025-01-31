### 1. Local Environment Setup
Install Python 3.11 and Create Virtual Environment
#### Upgrade pip
pip install --upgrade pip
#### Create a virtual environment
python3.11 -m venv myenv
#### Activate the virtual environment
##### Linux/macOS
source myenv/bin/activate
##### Windows
myenv\Scripts\activate

#### Install necessary packages
pip install fsspec pypdf neo4j graphdatascience openai langchain_ollama langchain_core langchain_community langchain_openai redis langchain streamlit

### 2. Docker Setup for Redis Container

#### Pull the Redis image
docker pull redis
#### Run the Redis container
docker run --name redis-container -p 6379:6379 -d redis

### 3. AWS AMI Setup
#### Connect to EC2 Instance

	1.	Instance Information
	2.	Locate your private key file (xxxx.pem).
	3.	Change the permissions of the key file (if necessary)
	4.	Connect to the EC2 instance using SSH

#### Install Dependencies on EC2
##### 3.1 Install Python 3.11
sudo dnf install python3.11
#### 3.2 Install Python pip
sudo yum -y install python-pip
#### 3.3 Transfer Files to EC2

#### 3.4 Set Up Python Environment on EC2

##### SSH into EC2 (if not already connected)
ssh -i xxx.pem ec2-user@ec2xxxxx.compute.amazonaws.com
##### Activate the virtual environment
source myenv/bin/activate
##### Install necessary packages
pip install fsspec pypdf neo4j graphdatascience openai langchain_ollama langchain_core langchain_community langchain_openai redis langchain streamlit
### Summary
This guide covers setting up a local Python environment, configuring Redis with Docker, connecting to an AWS EC2 instance, and setting up the necessary dependencies in a virtual environment.
