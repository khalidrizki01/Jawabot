FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    telnet \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U -r requirements.txt

COPY . .