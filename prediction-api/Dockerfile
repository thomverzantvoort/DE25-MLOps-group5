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

# ✅ Copy trained models from the parent 'models' folder on your host
COPY ../models ./models

# ---------- Expose Port ----------
EXPOSE 5000

# ---------- Environment Variables ----------
# MODEL_REPO points to where the models are located inside the container
ENV MODEL_REPO=/usr/src/myapp/models

# ---------- Run Application ----------
CMD ["python3", "app.py"]

# Optionally, could use ENTRYPOINT/CMD combo
# ENTRYPOINT ["python3"]
# CMD ["app.py"]
