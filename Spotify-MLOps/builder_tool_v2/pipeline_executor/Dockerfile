# ---------- Base Image ----------
FROM python:3.10.7-slim

# ---------- Copy Files ----------
# Copy the Python script into /home inside the container
ADD pipeline_executor.py /home/pipeline_executor.py

# Copy and install Python dependencies
ADD requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Entry Point ----------
ENTRYPOINT ["python", "/home/pipeline_executor.py"]
