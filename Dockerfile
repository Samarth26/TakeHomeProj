FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY census-bureau.data .
COPY census-bureau.columns .
COPY pipeline.py .
COPY segmentation.py .

CMD ["sh", "-c", "python pipeline.py && python segmentation.py"]
