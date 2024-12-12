# Training Analytics

## Installation (You may skip this step if you don't want to use it locally)

1. Install Python 3 if you haven't done so already.
   - Go to [python.org](https://python.org) for downloads

2. Create a virtual environment:
   - Choose a location in which to create the virtual environment. 
   - Note: We call it "venv" and don't put it in the repo (it's in .gitignore)
   ```bash
   cd <location for your virtual environment folder>
   python3 -m venv venv
   ```

3. Activate your virtual environment using the appropriate command for your shell:
   - MAC zsh: `source venv/bin/activate`
   - MS cmd.exe: `venv\Scripts\activate.bat`
   - Linux csh: `source venv/bin/activate.csh`
   
   For other shells, see [Python venv documentation](https://docs.python.org/3/library/venv.html)

4. Install Django:
   - Upgrade pip to the most current version:
     ```bash
     python3 -m pip install --upgrade pip
     ```
   - Then install Django:
     ```bash
     python3 -m pip install django
     ```

5. Set OPENAI_API_KEY as environment variable. Visit [https://platform.openai.com/api-keys] to create the API key.

6. Run the application:
   ```bash
   cd capstone
   python3 manage.py migrate
   python3 manage.py runserver
   ```

7. Visit [http://localhost:8000](http://localhost:8000) and verify that the application is working.

## Deployment

1. Install Docker if you haven't already:
   - Go to [docker.com](https://www.docker.com/products/docker-desktop) for downloads

2. Build the Docker image locally:
   ```bash
   docker build --platform linux/amd64 -t <APP_NAME>:latest .
   ```

3. Configure AWS credentials:
   - Create an AWS account if you don't have one
   - Install AWS CLI: [AWS CLI Installation Guide](https://aws.amazon.com/cli/)
   - Configure AWS credentials:
     ```bash
     aws configure
     ```

4. Tag and push the image to Amazon ECR:
   ```bash
   docker tag <APP_NAME>:latest <ECR_URL>/<APP_NAME>:latest
   docker push <ECR_URL>/<APP_NAME>:latest
   ```

5. On your EC2 instance, clean up existing containers:
   ```bash
   docker stop $(docker ps -q)
   docker rm $(docker ps -aq)
   ```

6. Pull and run the new container:
   ```bash
   # Pull the latest image
   docker pull <ECR_URL>/<APP_NAME>:latest

   # Run the container with environment variables
   docker run -d -p 8000:8000 \
     -e OPENAI_API_KEY="<your-api-key-here>" \
     <ECR_URL>/<APP_NAME>:latest
   ```

7. Run migrations in the new container:
   ```bash
   docker exec -it $(docker ps -q) python manage.py migrate
   ```

8. Visit your EC2 instance's public URL on port 8000 to verify the deployment:
   - Example: `http://<your-ec2-instance-ip>:8000`

Note: Replace the following placeholders with your actual values:
- `<APP_NAME>`: Your application name
- `<ECR_URL>`: Your Amazon ECR repository URL
- `<your-api-key-here>`: Your OpenAI API key
- `<your-ec2-instance-ip>`: Your EC2 instance's public IP address