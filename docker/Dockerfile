# GPU 딥러닝 + PySpark 실험 환경
# CUDA 12.1 런타임: NVIDIA 드라이버 530 이상이면 동작. 드라이버가 낮으면 BASE 를 cu118 이미지로 교체.
#   docker build --build-arg BASE=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime -t ami-dl:0.1 docker/
ARG BASE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Seoul \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# openjdk: PySpark 실행용, fonts-nanum: matplotlib 한글 그림
RUN apt-get update && apt-get install -y --no-install-recommends \
        git tzdata fonts-nanum openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /workspace
CMD ["sleep", "infinity"]
