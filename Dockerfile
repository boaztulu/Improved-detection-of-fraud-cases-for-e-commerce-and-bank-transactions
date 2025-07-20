# Use an official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port your Dash app will run on (default 8080 in your script)
EXPOSE 8080

# Run the dashboard app as the entry point
CMD ["python", "dashboard/dashboard_app.py"]
