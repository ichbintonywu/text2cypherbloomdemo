# Use the official Python image as the base
FROM python:3.9

# Set the working directory to /app
RUN mkdir -p /app
WORKDIR /app
COPY . ./

# Copy the requirements file and install the dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the app code
COPY secrets.toml /app/.streamlit/secrets.toml

# Expose the port that Streamlit uses
EXPOSE 80

# Define the entrypoint and command to run the app
CMD ["streamlit", "run", "text2cypherbloom.py", "--server.port=80"]
