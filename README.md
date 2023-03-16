# ImageClassifierAPI
PyTorch based image classifier API

## Build docker image
- From main dir run:
`docker build -f docker/Dockerfile . -t docker-django-v1.0`

## Run docker image
- From main dir run:
`docker run -p 8000:8000 docker-django-v1.0`

## Test connection from outside
- Run this command from terminal:
`curl 127.0.0.1:8000/api/status`