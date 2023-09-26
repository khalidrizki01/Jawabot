FROM node:18.17.1-slim

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# RUN npm run build
EXPOSE 5173
