FROM python:3.10-slim
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline_executor.py .

ENTRYPOINT ["python", "/workspace/pipeline_executor.py"]
