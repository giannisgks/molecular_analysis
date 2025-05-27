# Lightweight base image
FROM python:3.11-slim

# Ορισμός του working directory μέσα στο container
WORKDIR /app

# Αντιγραφή του αρχείου requirements.txt στο container
COPY requirements.txt .

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Εγκατάσταση των εξαρτήσεων της εφαρμογής
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή του φακέλου της εφαρμογής
COPY app/ ./app

# Δήλωση θύρας (default port of streamlit)
EXPOSE 8501

# Εκκίνηση της εφαρμογής
CMD ["streamlit", "run", "app/main.py"]
