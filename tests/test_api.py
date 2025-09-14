import os
import importlib
from fastapi.testclient import TestClient

def setup_module():
    # ensure the app reads safe env during import
    os.environ.setdefault("GAN_API_KEYS", "testkey")
    # import app after env set
    import inference.app as appmod  # noqa: F401
    importlib.reload(appmod)

def test_ping():
    from inference.app import app
    client = TestClient(app)
    resp = client.get("/ping", headers={"X-API-Key": "testkey"})
    assert resp.status_code == 200
    j = resp.json()
    assert "status" in j and j["status"] == "ok"

def test_metrics():
    from inference.app import app
    client = TestClient(app)
    resp = client.get("/metrics", headers={"X-API-Key": "testkey"})
    assert resp.status_code == 200
    text = resp.text
    assert "python_gc_objects_collected_total" in text