# Use an official Python runtime as the parent image
FROM pytorch/pytorch

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY MovieGenre.zip /app

# Install UNZIP
RUN apt-get update && apt-get install -y unzip && unzip MovieGenre.zip -d data && rm MovieGenre.zip

# Launch training
CMD ["python", "code/train.py"]