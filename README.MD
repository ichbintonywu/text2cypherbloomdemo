

# installation 
pip install --upgrade pip
python3.11 -m venv myenv
source myenv/bin/activate
pip install fsspec pypdf neo4j graphdatascience openai langchain_ollama langchain_core langchain_community langchain_openai redis langchain streamlit

# configuration of redis container
docker pull redis
docker run --name redis-container -p 6379:6379 -d redis


# on aws ami

######
Instance ID
i-0d884231d893caa1a (tonywudemo)
Open an SSH client.

Locate your private key file. The key used to launch this instance is tonywudemo.pem

Run this command, if necessary, to ensure your key is not publicly viewable.
chmod 400 "tonywudemo.pem"

Connect to your instance using its Public DNS:

ec2-3-107-95-58.ap-southeast-2.compute.amazonaws.com

Example:
ssh -i "tonywudemo.pem" ec2-user@ec2-3-107-95-58.ap-southeast-2.compute.amazonaws.com
######
sudo dnf install python3.11 
sudo yum -y install python-pip

ssh -i tonywudemo.pem ec2-user@ec2-3-107-95-58.ap-southeast-2.compute.amazonaws.com
scp -i tonywudemo.pem  /Users/tony/Documents/dev/neo4j/openai/text2cypherbloomdemo.zip ec2-user@ec2-3-107-95-58.ap-southeast-2.compute.amazonaws.com:/tmp 

source myenv/bin/activate
pip install fsspec pypdf neo4j graphdatascience openai langchain_ollama langchain_core langchain_community langchain_openai redis langchain streamlit