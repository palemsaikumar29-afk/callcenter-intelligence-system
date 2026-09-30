"""Call Center Intelligence System — entrypoint.

Serves the Gradio UI at http://localhost:7860 (CC_HOST/CC_PORT to change).
The Whisper model is loaded ONCE here at startup as a module-level singleton
— never inside a request handler. All secrets come from the environment.
"""

from __future__ import annotations

import logging

from src.agents import nodes
from src.database.db import CallCenterDB
from src.services import transcription as transcription_svc
from src.services.config import settings
from src.ui.app import build_app

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("callcenter")


def main() -> None:
    db = CallCenterDB(settings.db_path)
    nodes.set_db(db)
    if settings.preload_model:
        try:
            transcription_svc.preload_model()
            log.info("whisper model preloaded")
        except RuntimeError as exc:
            log.warning("whisper preload failed (transcription will error): %s", exc)
    build_app(db).launch(
        server_name=settings.app_host, server_port=settings.app_port, share=False
    )


if __name__ == "__main__":
    main()
