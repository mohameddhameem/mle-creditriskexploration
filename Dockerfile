# Use the official Apache Airflow image with Python 3.11
FROM apache/airflow:2.9.2-python3.11

# Switch to root to install system dependencies
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps, bash, build-essential, and cmake
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash build-essential cmake && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /opt/airflow

# Copy requirements file
COPY requirements.txt ./

# Switch back to airflow user
USER airflow

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container, ensuring correct ownership
COPY --chown=airflow:root . /opt/airflow
