## How to Use

1. Navigate to the `part1` folder:

   ```bash
   cd part1
   ```

2. Start the application using Docker Compose:

   ```bash
   docker-compose up
   ```

3. Open your browser and visit the following URL: [http://localhost:7860/](http://localhost:7860/)

4. Once you're done using the application, shut down the services:

   ```bash
   docker-compose down
   ```

5. Optionally, clean up unused Docker images:

   ```bash
   docker rmi part1-flask-api-service part1-gradio-app-service
   ```

---

## How to Train (Optional)

Training is not required for regular use. However, if you wish to train the models:

1. Navigate to the `part1` folder:

   ```bash
   cd part1
   ```

2. Build the Docker image for the training environment:

   ```bash
   docker build -t aif-project-image -f .\Dockerfile .
   ```

3. Run the training container:

   ```bash
   docker run --gpus all -v .\code:/app/code --ipc host --name aif-project-container aif-project-image
   ```

   **Note:** This step trains both the classifier and the anomaly detector (SVM).

