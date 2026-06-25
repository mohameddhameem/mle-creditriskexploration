FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
    procps \
    bash \
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /opt/airflow

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel

RUN grep -v "^phik" requirements.txt > requirements_clean.txt

RUN pip install --no-cache-dir -r requirements_clean.txt

RUN pip install --no-cache-dir jupyterlab streamlit uvicorn fastapi requests

COPY . /opt/airflow
