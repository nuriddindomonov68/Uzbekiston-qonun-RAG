import json
from typing import List, Dict, Any, Optional
from app.database.connection import get_db_connection


class SessionRepository:
    @staticmethod
    def create(session_id: str, settings: Dict[str, Any]) -> None:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, settings) VALUES (?, ?);",
                (session_id, json.dumps(settings)),
            )

    @staticmethod
    def get(session_id: str) -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?;", (session_id,)).fetchone()
        if row:
            return dict(id=row["id"], created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        settings=json.loads(row["settings"] or "{}"))
        return None

    @staticmethod
    def list_all() -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC;"
            ).fetchall()
        return [dict(id=r["id"], created_at=r["created_at"],
                     updated_at=r["updated_at"],
                     settings=json.loads(r["settings"] or "{}")) for r in rows]

    @staticmethod
    def touch(session_id: str) -> None:
        with get_db_connection() as conn:
            conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?;", (session_id,))

    @staticmethod
    def delete(session_id: str) -> None:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?;", (session_id,))


class ConversationRepository:
    @staticmethod
    def add_message(session_id: str, role: str, content: str) -> None:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?);",
                (session_id, role, content),
            )

    @staticmethod
    def get_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM conversations "
                "WHERE session_id = ? ORDER BY timestamp ASC, id ASC LIMIT ?;",
                (session_id, limit),
            ).fetchall()
        return [dict(role=r["role"], content=r["content"], timestamp=r["timestamp"]) for r in rows]

    @staticmethod
    def get_recent(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Oxirgi `limit` ta xabarni (eskidan yangiga) qaytaradi."""
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM ("
                "  SELECT id, role, content, timestamp FROM conversations "
                "  WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC;",
                (session_id, limit),
            ).fetchall()
        return [dict(role=r["role"], content=r["content"], timestamp=r["timestamp"]) for r in rows]

    @staticmethod
    def clear_history(session_id: str) -> None:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?;", (session_id,))


class DatasetRepository:
    @staticmethod
    def register(dataset_id: str, session_id: str, filename: str, filepath: str,
                 file_type: str, row_count: int, column_count: int,
                 schema: Dict[str, Any]) -> None:
        with get_db_connection() as conn:
            conn.execute(
                """INSERT INTO datasets
                   (id, session_id, filename, filepath, file_type, row_count, column_count, schema_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?);""",
                (dataset_id, session_id, filename, filepath, file_type,
                 row_count, column_count, json.dumps(schema)),
            )

    @staticmethod
    def get(dataset_id: str) -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?;", (dataset_id,)).fetchone()
        if row:
            return dict(id=row["id"], session_id=row["session_id"], filename=row["filename"],
                        filepath=row["filepath"], file_type=row["file_type"],
                        row_count=row["row_count"], column_count=row["column_count"],
                        schema=json.loads(row["schema_json"]), uploaded_at=row["uploaded_at"])
        return None

    @staticmethod
    def get_by_session(session_id: str) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets WHERE session_id = ? ORDER BY uploaded_at DESC, rowid DESC;",
                (session_id,),
            ).fetchall()
        return [dict(id=r["id"], session_id=r["session_id"], filename=r["filename"],
                     filepath=r["filepath"], file_type=r["file_type"],
                     row_count=r["row_count"], column_count=r["column_count"],
                     schema=json.loads(r["schema_json"]), uploaded_at=r["uploaded_at"]) for r in rows]

    @staticmethod
    def delete(dataset_id: str) -> None:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM datasets WHERE id = ?;", (dataset_id,))


class ModelRepository:
    @staticmethod
    def register(model_id: str, session_id: str, dataset_id: str, model_type: str,
                 algorithm: str, target_column: Optional[str], features: List[str],
                 metrics: Dict[str, Any], filepath: str) -> None:
        with get_db_connection() as conn:
            conn.execute(
                """INSERT INTO models
                   (id, session_id, dataset_id, model_type, algorithm, target_column, features, metrics, filepath)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (model_id, session_id, dataset_id, model_type, algorithm,
                 target_column, json.dumps(features), json.dumps(metrics), filepath),
            )

    @staticmethod
    def get(model_id: str) -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?;", (model_id,)).fetchone()
        if row:
            return dict(id=row["id"], session_id=row["session_id"], dataset_id=row["dataset_id"],
                        model_type=row["model_type"], algorithm=row["algorithm"],
                        target_column=row["target_column"],
                        features=json.loads(row["features"]),
                        metrics=json.loads(row["metrics"]),
                        filepath=row["filepath"], trained_at=row["trained_at"])
        return None

    @staticmethod
    def get_by_session(session_id: str) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM models WHERE session_id = ? ORDER BY trained_at DESC, rowid DESC;",
                (session_id,),
            ).fetchall()
        return [dict(id=r["id"], session_id=r["session_id"], dataset_id=r["dataset_id"],
                     model_type=r["model_type"], algorithm=r["algorithm"],
                     target_column=r["target_column"],
                     features=json.loads(r["features"]),
                     metrics=json.loads(r["metrics"]),
                     filepath=r["filepath"], trained_at=r["trained_at"]) for r in rows]


class ChartRepository:
    @staticmethod
    def register(chart_id: str, session_id: str, dataset_id: Optional[str],
                 title: Optional[str], chart_type: str, filepath: str) -> None:
        with get_db_connection() as conn:
            conn.execute(
                """INSERT INTO charts (id, session_id, dataset_id, title, chart_type, filepath)
                   VALUES (?, ?, ?, ?, ?, ?);""",
                (chart_id, session_id, dataset_id, title, chart_type, filepath),
            )

    @staticmethod
    def get_by_session(session_id: str) -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM charts WHERE session_id = ? ORDER BY created_at DESC, rowid DESC;",
                (session_id,),
            ).fetchall()
        return [dict(id=r["id"], session_id=r["session_id"], dataset_id=r["dataset_id"],
                     title=r["title"], chart_type=r["chart_type"],
                     filepath=r["filepath"], created_at=r["created_at"]) for r in rows]

    @staticmethod
    def get(chart_id: str) -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            r = conn.execute("SELECT * FROM charts WHERE id = ?;", (chart_id,)).fetchone()
        if r:
            return dict(id=r["id"], session_id=r["session_id"], dataset_id=r["dataset_id"],
                        title=r["title"], chart_type=r["chart_type"],
                        filepath=r["filepath"], created_at=r["created_at"])
        return None
