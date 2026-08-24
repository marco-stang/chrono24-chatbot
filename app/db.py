"""Gemeinsame sqlite-vec-Verbindung fuer app.retrieval und pipeline.index."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


def connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db
