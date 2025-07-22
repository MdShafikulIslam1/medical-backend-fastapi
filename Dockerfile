FROM python:3.10-slim-buster

# ওয়ার্কডির
WORKDIR /app

# সব ফাইল কপি করো
COPY . /app

# ডিপেন্ডেন্সি ইন্সটল করো
RUN pip install --no-cache-dir -r requirements.txt

# অ্যাপ রান করো
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

