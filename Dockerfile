FROM ultralytics/ultralytics:latest

WORKDIR /app

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости для YOLO и TensorRT заранее!
RUN pip install --no-cache-dir lap tensorrt redis opencv-python

# Копируем всё (включая папку models и файл newArch.py)
COPY . .

# Отключаем авто-обновления, чтобы он не лез в сеть
ENV ULTRALYTICS_OFFLINE=True

CMD ["python", "worker.py"]