"""App startup: `python app.py` serves HTTP on the configured port."""

import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

gradio = pytest.importorskip("gradio")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_for_port(port: int, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def test_app_serves_on_port(tmp_path):
    port = 17861
    db_path = str(tmp_path / "calls.db")
    env = dict(
        os.environ,
        CC_PORT=str(port),
        CC_DB_PATH=db_path,
        CC_PRELOAD_MODEL="0",  # skip weight download in this test
        PYTHONPATH=ROOT,
    )
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert _wait_for_port(port), "app did not start listening in time"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        assert resp.status == 200
        assert "gradio" in body.lower()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
