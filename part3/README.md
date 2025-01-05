How to use ?
1. Go to the `part3` folder.
2. Run the following command: `docker-compose up`
Then run `http://localhost:7860/`in your browser.
Once you're done, use `docker-compose down` and remove the unused docker images (`docker images`).

How to build the databases (not needed) ?
1. Go to the `part3` folder.
2. Run: `docker build -t aif-project-image -f .\Dockerfile .`
3. And then, run: `docker run -v .\code:/app/code --ipc host --name aif-project-container aif-project-image`
Notice that now you have two more files in the `code` folder which can be used to make the recommandation.