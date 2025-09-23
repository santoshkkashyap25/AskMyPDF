FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential poppler-utils git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt waitress

RUN python -m nltk.downloader punkt

COPY ./offline_cache/tinyroberta ./offline_cache/tinyroberta
COPY ./offline_cache/glove-twitter-25 ./offline_cache/glove-twitter-25

COPY . .

EXPOSE 5000
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]
