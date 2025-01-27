## How to Use

1. Navigate to the `part3` folder:

   ```bash
   cd part3
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
   docker rmi part3-flask-api-service part3-gradio-app-service
   ```
   
---

## How to Build Databases (Optional)

Training is not required for regular use. However, if you wish to train the models:

1. Navigate to the `part3` folder:

   ```bash
   cd part3
   ```

2. Build the Docker image for the training environment:

   ```bash
   docker build -t aif-project-image -f .\Dockerfile .
   ```

3. Run the training container:

   ```bash
   docker run -v .\code:/app/code --ipc host --name aif-project-container aif-project-image
   ```

Notice that now you have two more files in the `code` folder which can be used to make the recommandation.