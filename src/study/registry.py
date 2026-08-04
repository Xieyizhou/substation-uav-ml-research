"""SQLite-backed idempotent research registry."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from src.ml.artifacts import canonical_json


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  data_hash TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
  model_id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  onnx_hash TEXT NOT NULL,
  dataset_id TEXT,
  parent_model TEXT,
  manifest_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS studies (
  study_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  candidate_model TEXT NOT NULL,
  champion_model TEXT,
  status TEXT NOT NULL,
  promoted INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  study_id TEXT NOT NULL REFERENCES studies(study_id),
  tier TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  map_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  seed INTEGER NOT NULL,
  condition TEXT NOT NULL,
  config_hash TEXT,
  model_hash TEXT,
  status TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  failure_reason TEXT,
  UNIQUE(study_id, tier, scenario_id, condition)
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  name TEXT NOT NULL,
  value REAL NOT NULL,
  unit TEXT,
  PRIMARY KEY(run_id, name)
);
CREATE TABLE IF NOT EXISTS artifacts (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT,
  PRIMARY KEY(run_id, kind, path)
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


class ResearchRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def register_dataset(self, manifest, path):
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO datasets VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                  path=excluded.path, data_hash=excluded.data_hash,
                  manifest_json=excluded.manifest_json
                """,
                (
                    manifest["dataset_id"],
                    str(Path(path).resolve()),
                    manifest["data_sha256"],
                    canonical_json(manifest),
                    _now(),
                ),
            )

    def register_model(self, manifest, path):
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                  path=excluded.path, onnx_hash=excluded.onnx_hash,
                  manifest_json=excluded.manifest_json
                """,
                (
                    manifest["model_id"],
                    str(Path(path).resolve()),
                    manifest["onnx_sha256"],
                    manifest.get("dataset_id"),
                    manifest.get("parent_model"),
                    canonical_json(manifest),
                    _now(),
                ),
            )

    def get_model(self, model_id):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM models WHERE model_id=?", (model_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown model: {model_id}")
        return dict(row)

    def create_study(self, name, candidate_model, champion_model=None):
        study_id = f"study-{uuid.uuid4().hex[:12]}"
        now = _now()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO studies VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (study_id, name, candidate_model, champion_model, "created", now, now),
            )
        return study_id

    def get_study(self, study_id):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM studies WHERE study_id=?", (study_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown study: {study_id}")
        return dict(row)

    def ensure_runs(self, study_id, tier, matrix, *, config_hash="", model_hash=""):
        self.get_study(study_id)
        with self.connection() as connection:
            for row in matrix:
                identity = (
                    f"{study_id}|{tier}|{row['scenario_id']}|{row['condition']}"
                )
                run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
                connection.execute(
                    """
                    INSERT OR IGNORE INTO runs
                    (run_id, study_id, tier, scenario_id, map_id, target_id, seed,
                     condition, config_hash, model_hash, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        run_id,
                        study_id,
                        tier,
                        row["scenario_id"],
                        row["map_id"],
                        row["target_id"],
                        int(row["seed"]),
                        row["condition"],
                        config_hash,
                        model_hash,
                    ),
                )
        return self.runs(study_id, tier=tier)

    def runs(self, study_id, *, tier=None, statuses=None):
        query, values = "SELECT * FROM runs WHERE study_id=?", [study_id]
        if tier:
            query += " AND tier=?"
            values.append(tier)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            values.extend(statuses)
        query += " ORDER BY tier, scenario_id, condition"
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(query, values)]

    def set_run_status(self, run_id, status, *, failure_reason=None):
        if status not in {"pending", "running", "completed", "failed", "blocked"}:
            raise ValueError(f"invalid run status: {status}")
        started = _now() if status == "running" else None
        finished = _now() if status in {"completed", "failed", "blocked"} else None
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE runs SET status=?, failure_reason=?,
                  started_at=COALESCE(?, started_at), finished_at=?
                WHERE run_id=?
                """,
                (status, failure_reason, started, finished, run_id),
            )

    def record_metrics(self, run_id, metrics):
        with self.connection() as connection:
            for name, value in metrics.items():
                if value is None or isinstance(value, (dict, list)):
                    continue
                connection.execute(
                    """
                    INSERT INTO metrics(run_id, name, value, unit) VALUES (?, ?, ?, NULL)
                    ON CONFLICT(run_id, name) DO UPDATE SET value=excluded.value
                    """,
                    (run_id, name, float(value)),
                )
        self.set_run_status(run_id, "completed")

    def record_artifact(self, run_id, kind, path, sha256=None):
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(run_id, kind, path, sha256)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, kind, path) DO UPDATE SET sha256=excluded.sha256
                """,
                (run_id, kind, str(path), sha256),
            )

    def run_metrics(self, study_id, tier=None):
        query = """
          SELECT r.*, m.name, m.value FROM runs r
          LEFT JOIN metrics m ON r.run_id=m.run_id WHERE r.study_id=?
        """
        values = [study_id]
        if tier:
            query += " AND r.tier=?"
            values.append(tier)
        rows = {}
        with self.connection() as connection:
            for row in connection.execute(query, values):
                item = rows.setdefault(row["run_id"], dict(row))
                item.setdefault("metrics", {})
                if row["name"] is not None:
                    item["metrics"][row["name"]] = row["value"]
        return list(rows.values())

    def reset_incomplete(self, study_id, tier=None):
        query = """
          UPDATE runs SET status='pending', failure_reason=NULL, finished_at=NULL
          WHERE study_id=? AND status IN ('failed', 'blocked', 'running')
        """
        values = [study_id]
        if tier:
            query += " AND tier=?"
            values.append(tier)
        with self.connection() as connection:
            connection.execute(query, values)

    def promote(self, study_id):
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE studies SET promoted=1, status='promoted', updated_at=?
                WHERE study_id=?
                """,
                (_now(), study_id),
            )
