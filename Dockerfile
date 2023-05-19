# syntax=docker/dockerfile:1

FROM ubuntu:20.04

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --set python /usr/bin/python3

COPY requirements.txt .

# Install all the dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Bundle app source
COPY  web_app/ ./

EXPOSE 5000
CMD flask run --host 0.0.0.0 --port 5000