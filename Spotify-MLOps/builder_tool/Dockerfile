# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Working Directory ----------
WORKDIR /workspace

# ---------- Copy and Install Dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Executor Script ----------
COPY pipeline_executor.py .

# ---------- Entry Point ----------
ENTRYPOINT ["python", "/workspace/pipeline_executor.py"]
