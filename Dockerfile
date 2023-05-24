# syntax=docker/dockerfile:1

FROM python:3.8-slim

WORKDIR /app

COPY requirements_docker.txt .

RUN pip install --upgrade pip && pip install -r requirements_docker.txt


# Bundle app source
COPY web_app /app/web_app

WORKDIR /app/web_app

EXPOSE 5000
CMD gunicorn -c gunicorn_config.py app:app