# ---------- Base Image ----------
FROM python:3.12.7-slim

# ---------- Working Directory ----------
WORKDIR /usr/src/myapp

# ---------- Copy Requirements ----------
COPY requirements.txt .

# ---------- Install Dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy All Files ----------
COPY . .

# ---------- Expose Port ----------
EXPOSE 5000

# ---------- Environment Variables ----------
# Set to point at your prediction API container or endpoint
ENV PREDICTOR_API=http://localhost:5000/predict/

# ---------- Run App ----------
CMD ["python3", "app.py"]
