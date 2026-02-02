FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose port
EXPOSE 7120

# Start app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7120"]
