# ============================================================
#  MotionStudio サーバー
#
#  モデル5本はイメージに焼き込む（変わらないものなので）。
#  だから、どのデータセンターの GPU でも動く。
#  手本はイメージに入れない。動くたびにディスクから取ってくる（handler.py）。
#  手本を足すのに、ここは一切触らない。
# ============================================================
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY_DIR=/comfyui \
    MODELS_DIR=/models \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ---------- モデル（約25GB） ----------
# 取得元はすべて Comfy-Org/Wan-Animate-2。1本ずつ別の層にしておく。
# 途中で失敗しても、取り終えた分はやり直さずに済む。
ARG HF=https://huggingface.co/Comfy-Org/Wan-Animate-2/resolve/main
RUN mkdir -p /models/diffusion_models && curl -fL --retry 5 --retry-delay 10 -o \
    /models/diffusion_models/wan_animate_2_int8_convrot.safetensors \
    $HF/diffusion_models/wan_animate_2_int8_convrot.safetensors
RUN mkdir -p /models/text_encoders && curl -fL --retry 5 --retry-delay 10 -o \
    /models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    $HF/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
RUN mkdir -p /models/clip_vision && curl -fL --retry 5 --retry-delay 10 -o \
    /models/clip_vision/clip_vision_h.safetensors \
    $HF/clip_vision/clip_vision_h.safetensors
RUN mkdir -p /models/loras && curl -fL --retry 5 --retry-delay 10 -o \
    /models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
    $HF/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
RUN mkdir -p /models/vae && curl -fL --retry 5 --retry-delay 10 -o \
    /models/vae/Wan2_1_VAE_bf16.safetensors \
    $HF/vae/Wan2_1_VAE_bf16.safetensors

# ---------- ComfyUI ----------
# 一度うまく動いた版に固定したくなったら、COMFY_COMMIT を渡して作り直す。
ARG COMFY_COMMIT=""
RUN git clone https://github.com/comfyanonymous/ComfyUI /comfyui \
    && cd /comfyui \
    && if [ -n "$COMFY_COMMIT" ]; then git checkout "$COMFY_COMMIT"; fi

WORKDIR /comfyui
# torch は CUDA 13.0 版。12.6 版には Blackwell 世代（RTX PRO 6000 など）の命令が入っておらず、
# 48 GB PRO / 96 GB PRO の多くのマシンで起動直後に落ちた。
# 13.0 版はドライバが CUDA 13.0 以上のマシンでしか動かないので、
# エンドポイントの Advanced → CUDA バージョンは「13.0 以上すべて」にすること。
RUN pip install --upgrade pip \
    && pip install torch==2.14.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 \
    && pip install -r requirements.txt

# Sage Attention は生成が failure になる原因として特定済み。入れない。
RUN pip uninstall -y sageattention 2>/dev/null || true

# ComfyUI に /models を読ませる
RUN printf 'motionstudio:\n  base_path: /models\n  diffusion_models: diffusion_models\n  text_encoders: text_encoders\n  clip_vision: clip_vision\n  loras: loras\n  vae: vae\n' \
    > /comfyui/extra_model_paths.yaml

# ---------- カスタムノード ----------
RUN cd /comfyui/custom_nodes \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
    && pip install -r ComfyUI-VideoHelperSuite/requirements.txt

# ---------- ワーカー ----------
RUN pip install runpod boto3
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
