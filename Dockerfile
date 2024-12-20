FROM python:3.10-slim

RUN apt-get update && apt-get install -y redis-server && apt-get clean

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["bash"]

# CMD redis-server --daemonize yes && python app.py