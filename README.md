# Clustering Tutorial Web App

A web application to learn clustering algorithms interactively.

## Requirements

- [Docker](https://www.docker.com/get-started)

## Setup

1. **Build the Docker image**:

    ```bash
    docker build -t flask-clustering-app .
    ```

2. **Run the container**:

    ```bash
    docker run -it -p 5000:5000 flask-clustering-app
    ```

3. **Open the app** in your browser:

    ```
    http://127.0.0.1:5000
    ```
