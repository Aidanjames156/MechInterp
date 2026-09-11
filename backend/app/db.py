"""SQLite persistence for prompt sets, probes and intervention runs.

Vectors are stored as JSON arrays of floats: 768 numbers per GPT-2-small probe,
which is small enough that readability beats a binary blob.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_sets (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    positive_prompts TEXT NOT NULL,
    negative_prompts TEXT NOT NULL,
    created_at       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS probes (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    prompt_set_id TEXT NOT NULL REFERENCES prompt_sets(id),
    layer         INTEGER NOT NULL,
    vector        TEXT NOT NULL,
    metadata      TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS intervention_runs (
    id                TEXT PRIMARY KEY,
    probe_id          TEXT NOT NULL REFERENCES probes(id),
    prompt            TEXT NOT NULL,
    mode              TEXT NOT NULL,
    coefficient       REAL NOT NULL,
    output_baseline   TEXT NOT NULL,
    output_intervened TEXT NOT NULL,
    activation_trace  TEXT NOT NULL,
    params            TEXT NOT NULL,
    created_at        REAL NOT NULL
);
"""


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------ prompt sets
    def create_prompt_set(self, name: str, positive: list[str], negative: list[str]) -> dict:
        row = {
            "id": _new_id(),
            "name": name,
            "positive_prompts": positive,
            "negative_prompts": negative,
            "created_at": time.time(),
        }
        with self._lock:
            self._conn.execute(
                "INSERT INTO prompt_sets VALUES (?, ?, ?, ?, ?)",
                (row["id"], name, json.dumps(positive), json.dumps(negative), row["created_at"]),
            )
            self._conn.commit()
        return row

    def get_prompt_set(self, prompt_set_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM prompt_sets WHERE id = ?", (prompt_set_id,)).fetchone()
        return self._prompt_set_row(r) if r else None

    def list_prompt_sets(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM prompt_sets ORDER BY created_at DESC").fetchall()
        return [self._prompt_set_row(r) for r in rows]

    @staticmethod
    def _prompt_set_row(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"],
            "name": r["name"],
            "positive_prompts": json.loads(r["positive_prompts"]),
            "negative_prompts": json.loads(r["negative_prompts"]),
            "created_at": r["created_at"],
        }

    # ----------------------------------------------------------------- probes
    def create_probe(
        self, name: str, prompt_set_id: str, layer: int, vector: list[float], metadata: dict
    ) -> dict:
        row = {
            "id": _new_id(),
            "name": name,
            "prompt_set_id": prompt_set_id,
            "layer": layer,
            "vector": vector,
            "metadata": metadata,
            "created_at": time.time(),
        }
        with self._lock:
            self._conn.execute(
                "INSERT INTO probes VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"], name, prompt_set_id, layer,
                    json.dumps(vector), json.dumps(metadata), row["created_at"],
                ),
            )
            self._conn.commit()
        return row

    def get_probe(self, probe_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM probes WHERE id = ?", (probe_id,)).fetchone()
        return self._probe_row(r, include_vector=True) if r else None

    def list_probes(self, include_vector: bool = False) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM probes ORDER BY created_at DESC").fetchall()
        return [self._probe_row(r, include_vector) for r in rows]

    def delete_probe(self, probe_id: str) -> bool:
        with self._lock:
            self._conn.execute("DELETE FROM intervention_runs WHERE probe_id = ?", (probe_id,))
            cur = self._conn.execute("DELETE FROM probes WHERE id = ?", (probe_id,))
            self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _probe_row(r: sqlite3.Row, include_vector: bool) -> dict:
        d = {
            "id": r["id"],
            "name": r["name"],
            "prompt_set_id": r["prompt_set_id"],
            "layer": r["layer"],
            "metadata": json.loads(r["metadata"]),
            "created_at": r["created_at"],
        }
        if include_vector:
            d["vector"] = json.loads(r["vector"])
        return d

    # ------------------------------------------------------------------- runs
    def create_run(
        self,
        probe_id: str,
        prompt: str,
        mode: str,
        coefficient: float,
        output_baseline: str,
        output_intervened: str,
        activation_trace: list[dict],
        params: dict,
    ) -> dict:
        row = {
            "id": _new_id(),
            "probe_id": probe_id,
            "prompt": prompt,
            "mode": mode,
            "coefficient": coefficient,
            "output_baseline": output_baseline,
            "output_intervened": output_intervened,
            "activation_trace": activation_trace,
            "params": params,
            "created_at": time.time(),
        }
        with self._lock:
            self._conn.execute(
                "INSERT INTO intervention_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"], probe_id, prompt, mode, coefficient, output_baseline, output_intervened,
                    json.dumps(activation_trace), json.dumps(params), row["created_at"],
                ),
            )
            self._conn.commit()
        return row

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM intervention_runs WHERE id = ?", (run_id,)).fetchone()
        return self._run_row(r) if r else None

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM intervention_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._run_row(r) for r in rows]

    @staticmethod
    def _run_row(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"],
            "probe_id": r["probe_id"],
            "prompt": r["prompt"],
            "mode": r["mode"],
            "coefficient": r["coefficient"],
            "output_baseline": r["output_baseline"],
            "output_intervened": r["output_intervened"],
            "activation_trace": json.loads(r["activation_trace"]),
            "params": json.loads(r["params"]),
            "created_at": r["created_at"],
        }
