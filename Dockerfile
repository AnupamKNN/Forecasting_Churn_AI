# Use official Python base image
FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed later for things like DVC, MLflow, etc.)
RUN apt-get update && apt-get install -y git curl && apt-get clean

# Copy dependency file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files
COPY . .

# Copy model and preprocessor files (both classification and regression)
COPY artifacts/classification_model.pkl artifacts/classification_model.pkl
COPY artifacts/classification_preprocessor.pkl artifacts/classification_preprocessor.pkl
COPY artifacts/regression_model.pkl artifacts/regression_model.pkl
COPY artifacts/regression_preprocessor.pkl artifacts/regression_preprocessor.pkl

# If you later add app.py or main.py for backend, expose appropriate port
EXPOSE 8000

# CMD placeholder — replace when backend app is ready (e.g., with FastAPI or Flask)
CMD ["python", "-c", "print('Container built successfully. Backend not yet implemented.')"]
