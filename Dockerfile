# Python 3.9 resmi imajını temel al
FROM python:3.9.13

# Çalışma dizini oluştur
WORKDIR /app

# Gerekli Python kütüphanelerini yüklemek için requirements.txt dosyasını kopyala
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını konteynere kopyala
COPY . .

# Model dosyasını konteynere kopyala
# Model dosyanızın Dockerfile ile aynı dizinde olduğundan emin olun
COPY model_04012024_original_label_synthetic.h5 ./

# Uygulamayı çalıştır
CMD ["python", "./route.py"]
