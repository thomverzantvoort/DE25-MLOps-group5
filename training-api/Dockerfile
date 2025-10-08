# ---------- Base Image ----------
FROM python:3.12.7-slim

# ---------- Working Directory ----------
WORKDIR /usr/src/trainapp

# ---------- Copy Requirements ----------
COPY requirements.txt .

# ---------- Install Dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Project Files ----------
COPY . .

# ---------- Expose Port ----------
EXPOSE 5000

# ---------- Environment Variables ----------
ENV MODEL_REPO=/usr/src/trainapp/models

# ---------- Command ----------
CMD ["python3", "app.py"]
