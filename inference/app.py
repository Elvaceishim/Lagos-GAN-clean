import os
import io
import uuid
import time
import threading
import logging
import json
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np

# Lazy import for torch when needed
MODEL_PATH = os.environ.get("GAN_TS_MODEL", "checkpoints/production/G_AB_epoch02_ts.pt")
DEVICE_STR = os.environ.get("GAN_DEVICE", "cpu")

# Auth & rate limit config
API_KEYS = os.environ.get("GAN_API_KEYS", "").strip()
API_KEYS = [k for k in API_KEYS.split(",") if k] if API_KEYS else []
RATE_LIMIT_PER_MIN = int(os.environ.get("GAN_RATE_PER_MIN", "60"))  # requests per minute per key/ip
RATE_LIMIT_BURST = int(os.environ.get("GAN_RATE_BURST", "10"))

# Structured logging setup (JSON)
from pythonjsonlogger import jsonlogger
logger = logging.getLogger("lagos_gan")
handler = logging.StreamHandler()
fmt = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s %(api_key)s')
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Prometheus metrics
from prometheus_client import Counter, Histogram, core, generate_latest, CONTENT_TYPE_LATEST

# ensure generate_latest and CONTENT_TYPE_LATEST are available for /metrics
def _get_or_register(creator, name, *args, **kwargs):
    """
    Create the metric if not already registered; otherwise return existing collector.
    Avoids ValueError on re-import during tests / reloads.
    """
    registry = core.REGISTRY
    names = getattr(registry, "_names_to_collectors", {})
    if name in names:
        return names[name]
    return creator(name, *args, **kwargs)

# metrics (create-or-reuse)
REQUESTS = _get_or_register(Counter, "lagos_requests_total", "Total requests", ["endpoint", "api_key"])
ERRORS = _get_or_register(Counter, "lagos_errors_total", "Total errors", ["endpoint", "api_key"])
REQUESTS_CREATED = _get_or_register(Counter, "lagos_requests_created", "Request create times", ["endpoint", "api_key"])
# register histogram and also expose it under the name LATENCY because middleware expects LATENCY
REQUEST_LATENCY = _get_or_register(Histogram, "lagos_request_latency_seconds", "Request latency seconds", ["endpoint", "api_key"])
LATENCY = REQUEST_LATENCY

app = FastAPI(title="Lagos-GAN Inference (secured)")

# mount (optional) static UI directory
if os.path.isdir("inference/static"):
    app.mount("/static", StaticFiles(directory="inference/static"), name="static")

# In-memory rate limiter store: {key_or_ip: [timestamps]}
_rate_store = {}
_rate_lock = threading.Lock()

# Model holder
_model_lock = threading.Lock()
_model = None
_img_size_default = int(os.environ.get("GAN_IMG_SIZE", "64"))

def _authorize(request: Request) -> Optional[str]:
    # check X-API-Key header
    api_key = request.headers.get("x-api-key")
    if api_key and api_key in API_KEYS:
        return api_key
    # fallback: no key => deny
    return None

def _rate_allow(key: str) -> bool:
    now = time.time()
    window = 60.0
    with _rate_lock:
        q = _rate_store.setdefault(key, [])
        # remove old entries
        while q and q[0] < now - window:
            q.pop(0)
        if len(q) < RATE_LIMIT_PER_MIN + RATE_LIMIT_BURST:
            q.append(now)
            return True
        return False

def load_model(path):
    global _model
    import torch
    device = torch.device(DEVICE_STR)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with _model_lock:
        _model = torch.jit.load(path, map_location=device)
        _model.eval()
    logger.info("model_loaded", extra={"request_id": "-", "api_key": "-"})
    return _model

def preprocess_image_bytes(b, img_size):
    import torch
    img = Image.open(io.BytesIO(b)).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    arr = np.asarray(img).astype("float32")
    arr = (arr / 127.5) - 1.0
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return tensor

def postprocess_tensor(tensor):
    t = tensor.detach().cpu().squeeze(0).clamp(-1,1)
    arr = ((t + 1.0) * 127.5).permute(1,2,0).numpy().astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

@app.middleware("http")
async def metrics_and_auth_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    endpoint = request.url.path
    api_key = request.headers.get("x-api-key", "anon")
    extra = {"request_id": request_id, "api_key": api_key}
    logger.info("request_start", extra=extra)
    start = time.time()

    # Whitelist unauthenticated endpoints (health, metrics, UI, static)
    WHITELIST = ("/ping", "/metrics", "/ui", "/static")
    if any(endpoint == p or endpoint.startswith(p + "/") for p in WHITELIST):
        try:
            resp = await call_next(request)
        except Exception:
            ERRORS.labels(endpoint, api_key).inc()
            logger.exception("handler_error", extra=extra)
            raise
        latency = time.time() - start
        REQUESTS.labels(endpoint, api_key).inc()
        LATENCY.labels(endpoint, api_key).observe(latency)
        logger.info("request_end", extra=dict(extra, latency=latency))
        return resp

    # Auth
    key = _authorize(request)
    if key is None:
        ERRORS.labels(endpoint, api_key).inc()
        logger.warning("auth_failed", extra=extra)
        return JSONResponse(status_code=401, content={"detail":"Missing/invalid X-API-Key"})

    # Rate limiting (per API key)
    if not _rate_allow(key):
        ERRORS.labels(endpoint, api_key).inc()
        logger.warning("rate_limited", extra=extra)
        return JSONResponse(status_code=429, content={"detail":"Rate limit exceeded"})

    try:
        resp = await call_next(request)
    except Exception:
        ERRORS.labels(endpoint, api_key).inc()
        logger.exception("handler_error", extra=extra)
        raise
    latency = time.time() - start
    REQUESTS.labels(endpoint, api_key).inc()
    LATENCY.labels(endpoint, api_key).observe(latency)
    logger.info("request_end", extra=dict(extra, latency=latency))
    return resp

@app.get("/ping")
def ping():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate")
async def generate(request: Request, file: UploadFile = File(...), img_size: int = Query(None, description="resize input to this size before infer")):
    # request-level logging extras
    request_id = uuid.uuid4().hex[:8]
    api_key = request.headers.get("x-api-key", "anon")
    extra = {"request_id": request_id, "api_key": api_key}
    data = await file.read()
    size = img_size or int(os.environ.get("GAN_IMG_SIZE", "64"))
    try:
        inp = preprocess_image_bytes(data, size)
    except Exception as e:
        ERRORS.labels("/generate", api_key).inc()
        logger.exception("invalid_image", extra=extra)
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if _model is None:
        try:
            load_model(MODEL_PATH)
        except Exception as e:
            ERRORS.labels("/generate", api_key).inc()
            logger.exception("model_load_failed", extra=extra)
            raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    import torch
    device = torch.device(DEVICE_STR)
    inp = inp.to(device)
    with _model_lock:
        with torch.no_grad():
            out = _model(inp)
    buf = postprocess_tensor(out)
    tmp_name = f"results/generated_{uuid.uuid4().hex}.png"
    os.makedirs(os.path.dirname(tmp_name), exist_ok=True)
    with open(tmp_name, "wb") as f:
        f.write(buf.getbuffer())
    logger.info("generated", extra=extra)
    return FileResponse(tmp_name, media_type="image/png", filename=os.path.basename(tmp_name))

# Simple browser UI (uploads to /generate). Files served from inference/static/index.html
@app.get("/ui")
def ui():
    html_path = "inference/static/index.html"
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, "r").read())
    return HTMLResponse("<html><body><h3>No UI installed. Create inference/static/index.html</h3></body></html>")
