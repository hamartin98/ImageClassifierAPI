# syntax = docker/dockerfile:experimental
FROM python:3.10-slim-bullseye
RUN --mount=type=cache,mode=0777,target=/root/.cache/pip pip install pyyaml

ENV WORK_DIR=/src
ENV CACHE_DIR=.pip_cache
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN mkdir -p $CACHE_DIR $WORK_DIR

WORKDIR $WORK_DIR

COPY . $WORK_DIR
COPY requirements.txt /requirements.txt

# Install dependencies for opencv
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install --cache-dir $CACHE_DIR -r /requirements.txt

EXPOSE 8000

CMD ["python", "src/api/classifierAPI/manage.py", "runserver", "0.0.0.0:8000", "--noreload"]