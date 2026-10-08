# ============================================================
#  MotionStudio サーバー
#
#  中身は ComfyUI と、それを動かすための部品だけ。軽い。
#  モデルと手本は、別に用意した「消えないディスク」に置く。
#  だからイメージを作り直すのは、プログラムを変えたときだけ。
#  手本を足すのに、ここは一切触らない。
# ============================================================
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY_DIR=/comfyui \
    VOLUME_DIR=/runpod-volume \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg wget ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ---------- ComfyUI ----------
# 一度うまく動いた版に固定したくなったら、COMFY_COMMIT を渡して作り直す。
ARG COMFY_COMMIT=""
RUN git clone https://github.com/comfyanonymous/ComfyUI /comfyui \
    && cd /comfyui \
    && if [ -n "$COMFY_COMMIT" ]; then git checkout "$COMFY_COMMIT"; fi

WORKDIR /comfyui
RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \
    && pip install -r requirements.txt

# Sage Attention は生成が failure になる原因として特定済み。入れない。
RUN pip uninstall -y sageattention 2>/dev/null || true

# ---------- カスタムノード ----------
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
    && pip install -r ComfyUI-VideoHelperSuite/requirements.txt

# ---------- ワーカー ----------
RUN pip install runpod
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
