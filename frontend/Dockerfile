FROM node:18.17.1-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \ 
    curl \ 
    iproute2 \ 
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install
COPY . .
# RUN npm run build
