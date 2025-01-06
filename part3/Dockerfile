# Use an official Python runtime as the parent image
FROM pytorch/pytorch

# Set the working directory in the container to /app
WORKDIR /app

# Copy the code in the container (not necessarly all...)
COPY . /app 

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y build-essential

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r base_requirements.txt

# Download datas using gdown
RUN gdown --id 1xCm2RqqkuvIN5v3VIvRJ07u74agWXgaa -O /app/movies_metadata.csv

# Launch the creation of the databases
CMD ["python", "code/utils.py"]
