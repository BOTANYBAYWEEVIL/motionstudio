# ============================================================
#  MotionStudio サーバーレス・ワーカー
#  リクエストが来たときだけ起動し、終われば止まる。
#  待っている間の課金はない。
#
#  モデルはイメージに焼き込んである（/models）。
#  手本は RunPod のディスク（EU-RO-1）に置き、S3 API で取ってくる。
#  だからこのワーカーは、どのデータセンターの GPU でも動く。
#  手本を足すときは、ディスクの 手本/ に動画を1本置くだけ。
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
#
#  エンドポイントの環境変数（RunPod の画面で入れる）
#    VOLUME_ID   手本が入っているディスクのID
#    DATACENTER  そのディスクの場所（例 EU-RO-1）
#    S3_USER     RunPod のユーザーID（user_ で始まる）
#    S3_KEY      S3 API キー（rps_ で始まる）
# ============================================================
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import runpod

COMFY_DIR  = os.environ.get('COMFY_DIR', '/comfyui')
MODELS_DIR = os.environ.get('MODELS_DIR', '/models')     # イメージに焼き込んだモデル
COMFY_API  = 'http://127.0.0.1:8188'

INPUT_DIR  = os.path.join(COMFY_DIR, 'input')
OUTPUT_DIR = os.path.join(COMFY_DIR, 'output')
LOG_PATH   = '/tmp/comfyui.log'

# 手本の置き場（ディスク上のフォルダ名）
SAMPLE_PREFIX = '手本/'

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
SAVE_NODE = '246'          # 生成結果を書き出す SaveVideo（GraphBuilder.saveVideoNode と同じ）

# 返す動画の上限。RunPod の /run は約10MB まで。base64 で 4/3 倍になるので 7MB。
MAX_RETURN_BYTES = int(os.environ.get('MAX_RETURN_BYTES', '7000000'))

# 生成の上限。ここを過ぎたら諦める（ワーカーが無限に課金され続けるのを防ぐ）
GENERATE_TIMEOUT = int(os.environ.get('GENERATE_TIMEOUT', '3600'))


# ------------------------------------------------------------
#  ComfyUI の起動
# ------------------------------------------------------------

_comfy_proc = None


def missing_models():
    return [f'{k}/{n}' for k, n in REQUIRED_MODELS
            if not os.path.isfile(os.path.join(MODELS_DIR, k, n))]


# ------------------------------------------------------------
#  手本をディスクから取ってくる
# ------------------------------------------------------------

_s3 = None


def s3():
    """RunPod のディスクに S3 API でつなぐ。設定が無ければ例外。"""
    global _s3
    if _s3 is not None:
        return _s3
    need = ['VOLUME_ID', 'DATACENTER', 'S3_USER', 'S3_KEY']
    lack = [k for k in need if not os.environ.get(k, '').strip()]
    if lack:
        raise RuntimeError('エンドポイントの環境変数が足りません: ' + ', '.join(lack))

    import boto3
    from botocore.config import Config
    dc = os.environ['DATACENTER'].strip()
    _s3 = boto3.client(
        's3',
        region_name=dc,
        endpoint_url=f'https://s3api-{dc.lower()}.runpod.io/',
        aws_access_key_id=os.environ['S3_USER'].strip(),
        aws_secret_access_key=os.environ['S3_KEY'].strip(),
        config=Config(signature_version='s3v4',
                      s3={'addressing_style': 'path'},
                      retries={'max_attempts': 5, 'mode': 'standard'},
                      connect_timeout=20, read_timeout=120))
    return _s3


def remote_samples():
    """ディスクの 手本/ にある動画。{名前: (キー, バイト数)}"""
    bucket = os.environ['VOLUME_ID'].strip()
    out = {}
    pages = s3().get_paginator('list_objects_v2').paginate(
        Bucket=bucket, Prefix=SAMPLE_PREFIX)
    for page in pages:
        for obj in page.get('Contents', []) or []:
            key = obj['Key']
            name = key[len(SAMPLE_PREFIX):]
            if '/' in name or not name.lower().endswith(VIDEO_EXT):
                continue                      # 下の階層と動画以外は見ない
            out[name] = (key, int(obj.get('Size', 0)))
    return out


def sync_samples():
    """手元に無い（または大きさが違う）手本だけ取ってくる。名前の一覧を返す。"""
    bucket = os.environ['VOLUME_ID'].strip()
    os.makedirs(INPUT_DIR, exist_ok=True)
    remote = remote_samples()
    for name, (key, size) in remote.items():
        local = os.path.join(INPUT_DIR, name)
        if os.path.isfile(local) and os.path.getsize(local) == size:
            continue
        tmp = local + '.part'
        s3().download_file(bucket, key, tmp)
        os.replace(tmp, local)
    return sorted(remote)


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


_probe_cache = {}


def list_videos(names):
    out = []
    for name in names:
        p = os.path.join(INPUT_DIR, name)
        if not os.path.isfile(p):
            continue
        k = (name, os.path.getsize(p))
        if k not in _probe_cache:
            _probe_cache[k] = probe(p)
        info = dict(_probe_cache[k])
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
                outputs = entry.get('outputs', {}) or {}
                # 生成結果は SaveVideo（246）が output フォルダに書いたものだけ。
                # 手本を読むノードも「input フォルダの手本」を出力に載せてくるので、
                # type が output 以外は絶対に拾わない（手本がそのまま返った不具合の原因）。
                def videos_of(node_out):
                    return [f for f in collect_files(node_out)
                            if (f.get('type') or 'output') == 'output'
                            and str(f.get('filename', '')).lower().endswith(
                                ('.mp4', '.webm', '.mov'))]
                main = videos_of(outputs.get(SAVE_NODE, {}))
                if main:
                    return main[0]
                rest = videos_of(outputs)
                if rest:
                    return rest[0]
                raise RuntimeError('生成は終わったが、出来上がった動画が見つかりません: '
                                   + ', '.join(outputs.keys()))
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


def fit_size(name, blob):
    """RunPod の受け渡し上限に収まるよう、必要なときだけ再圧縮する。

    上限内ならそのまま返す。超えるときは、尺から逆算したビットレートで
    2パス圧縮する（大きさを狙い通りに収めやすい）。まだ超えたら絞って再挑戦。
    """
    if len(blob) <= MAX_RETURN_BYTES:
        return name, blob

    work = '/tmp/fit'
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    src = os.path.join(work, 'in' + (os.path.splitext(name)[1] or '.mp4'))
    with open(src, 'wb') as f:
        f.write(blob)

    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', src], capture_output=True, text=True)
    try:
        dur = float(r.stdout.strip())
    except ValueError:
        raise RuntimeError('出来上がった動画の長さを測れませんでした')
    has_audio = bool(subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries',
         'stream=index', '-of', 'csv=p=0', src],
        capture_output=True, text=True).stdout.strip())

    audio_bps = 96_000 if has_audio else 0
    dst = os.path.join(work, 'out.mp4')
    for ratio in (0.92, 0.80, 0.65):
        video_bps = int(MAX_RETURN_BYTES * 8 * ratio / dur) - audio_bps
        if video_bps < 150_000:
            raise RuntimeError('動画が長すぎて、返せる大きさに収まりません')
        common = ['-c:v', 'libx264', '-preset', 'medium', '-b:v', str(video_bps),
                  '-pix_fmt', 'yuv420p']
        p1 = ['ffmpeg', '-y', '-v', 'error', '-i', src, '-map', '0:v:0'] + common + \
             ['-pass', '1', '-passlogfile', os.path.join(work, 'x'),
              '-an', '-f', 'mp4', '/dev/null']
        p2 = ['ffmpeg', '-y', '-v', 'error', '-i', src, '-map', '0:v:0'] + \
             (['-map', '0:a:0', '-c:a', 'aac', '-b:a', str(audio_bps)] if has_audio else ['-an']) + \
             common + ['-pass', '2', '-passlogfile', os.path.join(work, 'x'),
                       '-movflags', '+faststart', dst]
        for cmd in (p1, p2):
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
            if r.returncode != 0:
                raise RuntimeError('再圧縮に失敗しました: ' + r.stderr[-800:])
        with open(dst, 'rb') as f:
            out = f.read()
        if len(out) <= MAX_RETURN_BYTES:
            return os.path.splitext(name)[0] + '.mp4', out
    raise RuntimeError('再圧縮しても、返せる大きさに収まりませんでした')


# ------------------------------------------------------------
#  窓口
# ------------------------------------------------------------

def handler(job):
    inp = job.get('input') or {}
    action = inp.get('action', 'generate')

    if action == 'ping':
        return {'ok': True}

    # 手本：ディスクから、手元に無い分だけ取ってくる
    try:
        names = sync_samples()
    except Exception as e:
        return {'error': f'手本を取ってこられませんでした: {e}'}

    if action == 'list':
        return {'videos': list_videos(names), 'chunk_frames': CHUNK_FRAMES}

    if action != 'generate':
        return {'error': f'知らない action です: {action}'}

    missing = missing_models()
    if missing:
        return {'error': 'モデルが揃っていません（イメージの作り直しが必要）',
                'missing': missing}

    if not ensure_comfy():
        return {'error': 'ComfyUI を起動できませんでした',
                'log': comfy_log_tail()}

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

    original_bytes = len(blob)
    try:
        name, blob = fit_size(name, blob)
    except Exception as e:
        return {'error': str(e)}

    return {
        'filename': name,
        'video_base64': base64.b64encode(blob).decode(),
        'bytes': len(blob),
        'original_bytes': original_bytes,
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
