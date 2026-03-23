"""
Watcher daemon - monitora la cartella documenti e indicizza i nuovi file
Usa watchdog per rilevare nuovi file e modifiche
"""

import logging
import os
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Aggiungi la directory corrente al path per import locali
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SUPPORTED_EXTENSIONS, WATCH_DELAY_SECONDS, WATCH_FOLDER
from db import init_db
from ingest_pipeline import importa_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class DocumentoHandler(FileSystemEventHandler):
    """Gestisce gli eventi filesystem e indicizza i nuovi file."""

    def __init__(self):
        self._in_attesa = {}  # file_path -> timestamp ultimo evento

    def on_created(self, event):
        if not event.is_directory:
            self._schedula(event.src_path, "creato")

    def on_modified(self, event):
        if not event.is_directory:
            self._schedula(event.src_path, "modificato")

    def on_moved(self, event):
        if not event.is_directory:
            self._schedula(event.dest_path, "spostato")

    def _schedula(self, path: str, motivo: str):
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return

        log.info(
            "File %s: %s — schedulo import tra %ds",
            motivo,
            os.path.basename(path),
            WATCH_DELAY_SECONDS,
        )
        self._in_attesa[path] = time.time()

    def processa_pendenti(self):
        ora = time.time()
        da_processare = [
            path for path, ts in self._in_attesa.items()
            if ora - ts >= WATCH_DELAY_SECONDS
        ]

        for path in da_processare:
            del self._in_attesa[path]

            if not os.path.exists(path):
                log.warning("File non trovato (cancellato?): %s", path)
                continue

            log.info("Processo: %s", os.path.basename(path))
            try:
                res = importa_file(path)

                if res["status"] == "importato":
                    extra = f" → {res['totale']:.2f} EUR" if res.get("totale") is not None else ""
                    log.info("  ✓ %s (%s)%s", res["file"], res["tipo"], extra)
                elif res["status"] == "già_importato":
                    log.info("  ~ già presente: %s", res["file"])
                else:
                    log.warning("  ✗ %s → %s", res["file"], res["status"])
            except Exception as e:
                log.error("  Errore su %s: %s", os.path.basename(path), e)


def main():
    cartella = WATCH_FOLDER

    if not os.path.isdir(cartella):
        log.error("Cartella non trovata: %s", cartella)
        sys.exit(1)

    init_db()

    log.info("Watcher avviato — monitoro: %s", cartella)
    log.info("Estensioni supportate: %s", ", ".join(sorted(SUPPORTED_EXTENSIONS)))

    handler = DocumentoHandler()
    observer = Observer()
    observer.schedule(handler, cartella, recursive=True)
    observer.start()

    try:
        while True:
            handler.processa_pendenti()
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Watcher fermato.")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
