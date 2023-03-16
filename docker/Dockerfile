# https://blog.logrocket.com/dockerizing-django-app/

# Run from main dir

FROM python:3.8

ENV WORK_DIR=/src
ENV CACHE_DIR=.pip_cache

#RUN cache restore

RUN mkdir -p $CACHE_DIR
RUN mkdir -p $WORK_DIR

WORKDIR $WORK_DIR

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . $WORK_DIR

RUN pip install --cache-dir $CACHE_DIR -r requirements.txt

EXPOSE 8000
#EXPOSE 8001

#CMD python src/api/classifierAPI/manage.py runserver --noreload
#CMD ["python", "src/api/classifierAPI/manage.py", "collectstatic"]

CMD ["python", "src/api/classifierAPI/manage.py", "runserver", "0.0.0.0:8000", "--noreload"]