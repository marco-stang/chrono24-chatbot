"""Tages-Token-Budget in SQLite — Deckel für die öffentliche Demo."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path


class TokenBudget:
    def __init__(self, db_path: Path, daily_limit: int):
        self.daily_limit = daily_limit
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS budget (day TEXT PRIMARY KEY, tokens INTEGER NOT NULL)"
        )
        self.conn.commit()

    def spend(self, tokens: int) -> None:
        self.conn.execute(
            "INSERT INTO budget (day, tokens) VALUES (?, ?) "
            "ON CONFLICT(day) DO UPDATE SET tokens = tokens + excluded.tokens",
            (date.today().isoformat(), tokens),  # noqa: DTZ011
        )
        self.conn.commit()

    def used_today(self) -> int:
        row = self.conn.execute(
            "SELECT tokens FROM budget WHERE day = ?", (date.today().isoformat(),)  # noqa: DTZ011
        ).fetchone()
        return row[0] if row else 0

    def remaining(self) -> int:
        return self.daily_limit - self.used_today()
