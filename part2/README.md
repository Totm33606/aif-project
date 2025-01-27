## How to Use (same as in part2)

1. Navigate to the `part2` folder:

   ```bash
   cd part2
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
   docker rmi part2-flask-api-service part2-gradio-app-service
   ```