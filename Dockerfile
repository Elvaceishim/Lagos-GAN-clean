FROM python:3.10-slim

WORKDIR /app

# minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.inference.txt .
# install runtime deps then PyTorch CPU wheels
RUN pip install --no-cache-dir -r requirements.inference.txt && \
    pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    apt-get purge -y --auto-remove gcc

COPY . .

ENV GAN_TS_MODEL=/app/checkpoints/production/G_AB_epoch02_ts.pt
ENV GAN_IMG_SIZE=64
# do NOT hardcode keys here
ENV GAN_API_KEYS=""

EXPOSE 8080
CMD ["uvicorn","inference.app:app","--host","0.0.0.0","--port","8080","--workers","1"]