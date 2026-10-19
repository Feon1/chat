FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["python", "server.py"]
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
