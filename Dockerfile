# Use the official PyTorch image
FROM pytorch/pytorch:latest

# Set a working directory
WORKDIR /app

# Copy requirements.txt into the container (if you have one)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python scripts (train.py, test.py) into the container
COPY train.py /app/train.py


# Copy the 'Data' folder into the container
COPY Data /app/Data

# Run train.py when the container starts
CMD ["python3", "/app/train.py"]
