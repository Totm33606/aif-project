# Use an official Python runtime as the parent image
FROM pytorch/pytorch

# Set the working directory in the container to /app
WORKDIR /app

# Copy the code inside the container 
COPY . /app

# Download gdown
RUN pip install --trusted-host pypi.python.org -r base_requirements.txt

# Download datas using gdown
RUN gdown --id 1TYyn6x6z-NxCFHjBm__8d0liNJM0i_g7 -O /app/MovieGenre.zip
RUN gdown --id 1yPLa-Da9ercWn4J43WW3iBhlXDDWLX8d -O /app/anomaly_data.zip

# Install UNZIP
RUN apt-get update && apt-get install -y unzip
RUN unzip MovieGenre.zip -d data && rm MovieGenre.zip
RUN unzip anomaly_data.zip -d anomaly && rm anomaly_data.zip

# Launch training
CMD ["python", "code/train.py"]