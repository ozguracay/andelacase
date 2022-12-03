FROM python:3.10.8
WORKDIR /app

COPY requirements.txt app/requirements.txt 
RUN pip install -r app/requirements.txt

COPY src app/src
COPY *.py app/
