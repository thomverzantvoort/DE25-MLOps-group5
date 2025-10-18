FROM python:3.10-slim
WORKDIR /usr/src/myapp

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline_executor.py .

ENTRYPOINT ["python", "pipeline_executor.py"]
