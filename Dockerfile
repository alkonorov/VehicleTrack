FROM python:3.11-slim

# Системные зависимости для OpenCV и ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir numpy==1.26.4
# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Создаём папки для монтирования
RUN mkdir -p /app/data /app/output

# По умолчанию — headless режим с сохранением в файл
CMD ["python", "main.py", "--no-display", "-v", "/app/data/test_video.mp4", "-o", "/app/output/result.mp4"]