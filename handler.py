# ============================================================
#  MotionStudio サーバーレス・ワーカー
#  リクエストが来たときだけ起動し、終われば止まる。
#  待っている間の課金はない。
#
#  モデルと手本は、消えないディスク（ネットワークボリューム）に置く。
#  手本を足すときは、そこに動画を1本置くだけ。
#  このプログラムもイメージも、触る必要はない。
#
#  窓口は1つ。input.action で仕事を分ける。
#
#    {"input": {"action": "list"}}
#        → 使える手本の一覧（フレーム数・fps・解像度・作れる尺）
#
#    {"input": {"action": "generate",
#               "prompt": <ComfyUI のグラフ>,
#               "images": [{"name": "ms_ref.jpg", "data": "<base64>"}]}}
#        → 生成した動画を base64 で返す
# ============================================================
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
import uuid

import runpod

COMFY_DIR  = os.environ.get('COMFY_DIR', '/comfyui')
VOLUME_DIR = os.environ.get('VOLUME_DIR', '/runpod-volume')
COMFY_API  = 'http://127.0.0.1:8188'

# 消えないディスク側の置き場。手本を足すのはここ。
VOL_MODELS = os.path.join(VOLUME_DIR, 'models')
VOL_INPUT  = os.path.join(VOLUME_DIR, '手本')

# ComfyUI から見た場所。中身は上のディスクへの入り口にすぎない。
INPUT_DIR  = os.path.join(COMFY_DIR, 'input')
MODELS_DIR = os.path.join(COMFY_DIR, 'models')
OUTPUT_DIR = os.path.join(COMFY_DIR, 'output')
LOG_PATH   = '/tmp/comfyui.log'

# ワークフローが実際に読むモデル。欠けていれば起動前に気づけるように。
REQUIRED_MODELS = [
    ('diffusion_models', 'wan_animate_2_int8_convrot.safetensors'),
    ('loras',            'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors'),
    ('text_encoders',    'umt5_xxl_fp8_e4m3fn_scaled.safetensors'),
    ('vae',              'Wan2_1_VAE_bf16.safetensors'),
    ('clip_vision',      'clip_vision_h.safetensors'),
]

VIDEO_EXT = ('.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v')
CHUNK_FRAMES = 81          # WanAnimate2ToVideo の length

# 生成の上限。ここを過ぎたら諦める（ワーカーが無限に課金され続けるのを防ぐ）
GENERATE_TIMEOUT = int(os.environ.get('GENERATE_TIMEOUT', '3600'))


# ------------------------------------------------------------
#  ComfyUI の起動
# ------------------------------------------------------------

_comfy_proc = None


def attach_volume():
    """消えないディスクを ComfyUI の置き場として使わせる。

    実体を複写せず、入り口（シンボリックリンク）を張るだけ。
    だから起動は速く、ディスクの中身を足せば即座に反映される。
    返り値は、足りないものの一覧。空なら準備完了。
    """
    missing = []

    if not os.path.isdir(VOLUME_DIR):
        return ['ディスクがつながっていません（エンドポイントにボリュームを割り当ててください）']

    os.makedirs(VOL_INPUT, exist_ok=True)

    # 手本
    if not os.path.islink(INPUT_DIR):
        if os.path.isdir(INPUT_DIR):
            shutil.rmtree(INPUT_DIR, ignore_errors=True)
        try:
            os.symlink(VOL_INPUT, INPUT_DIR)
        except FileExistsError:
            pass

    # モデル（種類ごとに入り口を張る。ComfyUI 側の他の階層は壊さない）
    if not os.path.isdir(VOL_MODELS):
        missing.append(f'{VOL_MODELS} がありません')
        return missing

    for kind, name in REQUIRED_MODELS:
        src = os.path.join(VOL_MODELS, kind)
        dst = os.path.join(MODELS_DIR, kind)
        os.makedirs(src, exist_ok=True)
        if not os.path.islink(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
        if not os.path.isfile(os.path.join(src, name)):
            missing.append(f'{kind}/{name}')

    return missing


def _comfy_up(timeout=2):
    try:
        urllib.request.urlopen(COMFY_API + '/system_stats', timeout=timeout)
        return True
    except Exception:
        return False


def ensure_comfy(wait=300):
    """ComfyUI を同じコンテナ内で動かす。すでに動いていれば何もしない。"""
    global _comfy_proc
    if _comfy_up():
        return True

    if _comfy_proc is None or _comfy_proc.poll() is not None:
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        _comfy_proc = subprocess.Popen(
            ['python', 'main.py',
             '--listen', '127.0.0.1', '--port', '8188',
             '--disable-auto-launch', '--disable-metadata'],
            cwd=COMFY_DIR,
            stdout=open(LOG_PATH, 'w'),
            stderr=subprocess.STDOUT,
            start_new_session=True)

    for _ in range(wait):
        if _comfy_proc.poll() is not None:
            return False
        if _comfy_up():
            return True
        time.sleep(1)
    return False


def comfy_log_tail(n=2000):
    try:
        return open(LOG_PATH, errors='replace').read()[-n:]
    except Exception:
        return ''


# ------------------------------------------------------------
#  手本の一覧
# ------------------------------------------------------------

def probe(path):
    """実フレーム数・fps・解像度を測る。アプリはこれで尺の上限を決める。"""
    def ff(args):
        r = subprocess.run(['ffprobe', '-v', 'error'] + args + [path],
                           capture_output=True, text=True)
        return r.stdout.strip()

    out = ff(['-select_streams', 'v:0',
              '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
              '-show_entries', 'format=duration',
              '-of', 'default=noprint_wrappers=1:nokey=0'])
    d = {}
    for line in out.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            d[k] = v

    def num(x):
        try:
            if '/' in str(x):
                a, b = str(x).split('/')
                return float(a) / float(b) if float(b) else 0.0
            return float(x)
        except Exception:
            return 0.0

    fps = num(d.get('r_frame_rate', 0))
    dur = num(d.get('duration', 0))
    frames = int(num(d.get('nb_frames', 0)))

    if frames <= 0:
        frames = int(num(ff(['-select_streams', 'v:0', '-count_frames',
                             '-show_entries', 'stream=nb_read_frames',
                             '-of', 'csv=p=0'])))
    if frames <= 0 and fps > 0 and dur > 0:
        frames = int(fps * dur)

    return {
        'frames': frames,
        'fps': round(fps, 3),
        'width': int(num(d.get('width', 0))),
        'height': int(num(d.get('height', 0))),
        'seconds': round(frames / fps, 2) if fps else 0,
        'max_chunks': frames // CHUNK_FRAMES,
    }


def list_videos():
    os.makedirs(INPUT_DIR, exist_ok=True)
    out = []
    for name in sorted(os.listdir(INPUT_DIR)):
        if not name.lower().endswith(VIDEO_EXT):
            continue
        p = os.path.join(INPUT_DIR, name)
        if not os.path.isfile(p):
            continue
        info = probe(p)
        info['name'] = name
        out.append(info)
    return out


# ------------------------------------------------------------
#  生成
# ------------------------------------------------------------

def write_images(images):
    """アプリから送られた画像を ComfyUI の input に置く。"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    written = []
    for item in images or []:
        name = os.path.basename(item.get('name') or f'ms_{uuid.uuid4().hex}.jpg')
        data = item.get('data') or ''
        if ',' in data[:64] and data[:5] in ('data:', 'data'):
            data = data.split(',', 1)[1]       # data: URL 形式も受ける
        with open(os.path.join(INPUT_DIR, name), 'wb') as f:
            f.write(base64.b64decode(data))
        written.append(name)
    return written


def post_prompt(prompt, client_id):
    body = json.dumps({'prompt': prompt, 'client_id': client_id}).encode()
    req = urllib.request.Request(COMFY_API + '/prompt', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)['prompt_id']


def collect_files(outputs):
    files = []

    def walk(x):
        if isinstance(x, list):
            for e in x:
                walk(e)
        elif isinstance(x, dict):
            if 'filename' in x:
                files.append(x)
            else:
                for v in x.values():
                    walk(v)
    walk(outputs)
    return files


def wait_for(prompt_id, deadline):
    """完了まで待つ。失敗していれば例外。"""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    COMFY_API + f'/history/{prompt_id}', timeout=30) as r:
                hist = json.load(r)
        except Exception:
            time.sleep(2)
            continue

        entry = hist.get(prompt_id)
        if entry:
            status = entry.get('status', {}) or {}
            if status.get('status_str') == 'error':
                raise RuntimeError(
                    '生成が失敗しました: ' + str(status.get('messages'))[:1500])
            if status.get('completed') or entry.get('outputs'):
                files = collect_files(entry.get('outputs', {}))
                videos = [f for f in files
                          if str(f.get('filename', '')).lower().endswith(
                              ('.mp4', '.webm', '.mov'))]
                if videos:
                    return videos[0]
                if files:
                    return files[0]
        time.sleep(2)
    raise TimeoutError(f'{GENERATE_TIMEOUT} 秒を過ぎても終わりませんでした')


def read_output(f):
    sub = f.get('subfolder') or ''
    name = f.get('filename')
    typ = f.get('type') or 'output'
    base = OUTPUT_DIR if typ == 'output' else os.path.join(COMFY_DIR, typ)
    path = os.path.join(base, sub, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'出力が見つかりません: {path}')
    with open(path, 'rb') as fp:
        return name, fp.read()


# ------------------------------------------------------------
#  窓口
# ------------------------------------------------------------

def handler(job):
    inp = job.get('input') or {}
    action = inp.get('action', 'generate')

    missing = attach_volume()
    if missing:
        return {'error': 'ディスクの準備ができていません',
                'missing': missing,
                'hint': f'{VOL_MODELS} にモデル5本、{VOL_INPUT} に手本の動画を置いてください'}

    if not ensure_comfy():
        return {'error': 'ComfyUI を起動できませんでした',
                'log': comfy_log_tail()}

    if action == 'ping':
        return {'ok': True}

    if action == 'list':
        return {'videos': list_videos(), 'chunk_frames': CHUNK_FRAMES}

    if action != 'generate':
        return {'error': f'知らない action です: {action}'}

    prompt = inp.get('prompt')
    if not isinstance(prompt, dict) or not prompt:
        return {'error': 'prompt がありません'}

    try:
        write_images(inp.get('images'))
    except Exception as e:
        return {'error': f'画像を置けませんでした: {e}'}

    started = time.time()
    try:
        prompt_id = post_prompt(prompt, inp.get('client_id') or uuid.uuid4().hex)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors='replace')[:1500]
        return {'error': f'グラフが拒否されました（{e.code}）', 'detail': detail}
    except Exception as e:
        return {'error': f'投入に失敗しました: {e}'}

    try:
        f = wait_for(prompt_id, started + GENERATE_TIMEOUT)
        name, blob = read_output(f)
    except Exception as e:
        return {'error': str(e), 'log': comfy_log_tail(1200)}

    return {
        'filename': name,
        'video_base64': base64.b64encode(blob).decode(),
        'bytes': len(blob),
        'seconds_taken': round(time.time() - started, 1),
        'prompt_id': prompt_id,
    }


# コンテナが起きた時点で ComfyUI を立ち上げておく。
# 最初のリクエストの待ち時間を、その分だけ短くする。
try:
    ensure_comfy(wait=5)
except Exception:
    pass

runpod.serverless.start({'handler': handler})
