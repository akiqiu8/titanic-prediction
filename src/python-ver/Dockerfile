# Use a stable base image
FROM python:3.11

# Set working directory inside container
WORKDIR /python-app

# Copy the dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY main.py .

# Run the script
CMD ["python", "main.py"]


# Under titanic-prediction directory, download Kaggle datasets in data file, build and run the Docker container:
# docker build -t titanic-pred-python  ./src/python-ver
# docker run --rm -v "$(pwd)/src/data:/python-app/data" titanic-pred-python
