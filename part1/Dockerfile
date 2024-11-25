# Use an official Python runtime as the parent image
FROM pytorch/pytorch

# Set the working directory in the container to /app
WORKDIR /app

# Download gdown
RUN pip install gdown==5.2.0

# Download datas using gdown
RUN gdown --id 1TYyn6x6z-NxCFHjBm__8d0liNJM0i_g7 -O /app/MovieGenre.zip

# Install UNZIP
RUN apt-get update && apt-get install -y unzip && unzip MovieGenre.zip -d data && rm MovieGenre.zip

# Launch training
CMD ["python", "code/train.py"]