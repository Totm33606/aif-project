How to use ?
1. Go to the `part1` folder.
2. Run the following command: `docker-compose up`
Then run `http://localhost:7860/`in your browser.
Once you're done, use `docker-compose down` and remove the unused docker images (`docker images`).

How to train (not needed) ?
1. Go to the `part1` folder.
2. Run: `docker build -t aif-project-image -f .\Dockerfile .`
3. And then, run: `docker run --gpus all -v .\code:/app/code --ipc host --name aif-project-container aif-project-image`
Note: Here you'll train the classifier **and** the anomaly detector (SVM).