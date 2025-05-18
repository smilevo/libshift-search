"""
db_handler.py

Database handler module for managing similarity score caching in SQLite.
Supports creation, querying, insertion, and cleanup of cached similarity scores
used in model-based comparisons across versions of APIs or libraries.

License:
This software is licensed strictly for non-commercial academic and research purposes.
Use is permitted only by individual researchers, students, and educators
affiliated with academic institutions, and only for scholarly work.

Prohibited Uses:
- Commercial use in any form, including but not limited to products, services, or for-profit research.
- Redistribution, sublicensing, or modification without prior written permission.
- Use by or integration into any large language models (LLMs), AI agents or systems, bots, or autonomous software
  whether for training, inference, benchmarking, or any other purpose.

By using this software, you agree to abide by these terms.

(c) 2025 Anush Krishna V (anushkrishnav). All rights reserved.

Author: Anush Krishna V  
Created: 1 May 2025
"""

import os
import sqlite3
from datetime import datetime
from src.config import Config


class DBHandler:
    """
    Handles creation, access, and maintenance of the SQLite database used to cache similarity scores.

    Attributes:
        config (Config): Configuration object holding database settings.
        conn (sqlite3.Connection): SQLite connection instance.
        cursor (sqlite3.Cursor): Cursor to execute SQL commands.
    """

    def __init__(self, config: Config):
        """
        Initializes the database handler and sets up the connection and schema.

        Args:
            config (Config): Configuration object with path and table name.
        """
        self.config = config
        self.path = config.CACHE_DB_PATH
        self.__clean_sqlite_locks(self.path)
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        self._connect()
        self._create_table()
        self._setup_weekly_trigger()

    def _create_table(self):
        """
        Creates the cache table if it does not already exist.
        """
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.config.CACHE_TABLE} (
            removed_id TEXT,
            snapshot_id TEXT,
            model_name TEXT,
            modality TEXT,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (removed_id, snapshot_id, model_name, modality)
        );
        """)
        self.conn.commit()

    def __clean_sqlite_locks(self, db_path: str):
        """
        Deletes SQLite lock-related files in the database directory.

        Args:
            db_path (str): Path to the SQLite DB file or directory.
        """
        print(f"Cleaning SQLite lock files in: {db_path}")
        if db_path.endswith(".db"):
            db_path = os.path.dirname(db_path)

        patterns = [".db-wal", ".db-shm", ".db-journal"]
        try:
            files = os.listdir(db_path)
            for file in files:
                if any(file.endswith(pattern) for pattern in patterns):
                    os.remove(os.path.join(db_path, file))
                    print(f"Removed lock file: {file}")
        except Exception as e:
            print(f"Error cleaning SQLite lock files: {e}")

    def _connect(self):
        """
        Sets SQLite to WAL mode for improved concurrent access behavior.
        """
        self.conn.execute("PRAGMA journal_mode=WAL;")

    def _setup_weekly_trigger(self):
        """
        Deletes cache entries older than 7 days to maintain a rolling cache.
        """
        self.cursor.execute(f"""
        DELETE FROM {self.config.CACHE_TABLE}
        WHERE created_at < DATETIME('now', '-7 days');
        """)
        self.conn.commit()

    def cache_it(self, removed_id, snapshot_id, model_name, modality, score):
        """
        Caches a single similarity score if it is not already present.

        Args:
            removed_id (str): Removed method/function ID.
            snapshot_id (str): Matching method/function ID in a snapshot version.
            model_name (str): Name of the model used.
            modality (str): Textual, structural, or other modality of comparison.
            score (float): Similarity score to cache.
        """
        self.cursor.execute(f"""
        SELECT 1 FROM {self.config.CACHE_TABLE}
        WHERE removed_id = ? AND snapshot_id = ? AND model_name = ? AND modality = ?
        """, (removed_id, snapshot_id, model_name, modality))
        
        if not self.cursor.fetchone():
            self.cursor.execute(f"""
            INSERT INTO {self.config.CACHE_TABLE}
            (removed_id, snapshot_id, model_name, modality, score)
            VALUES (?, ?, ?, ?, ?)
            """, (removed_id, snapshot_id, model_name, modality, score))
            self.conn.commit()

    def cache_many(self, entries):
        """
        Batch insert similarity scores if they are not already cached.

        Args:
            entries (list of tuple): Each entry should be 
                (removed_id, snapshot_id, model_name, modality, score).
        """
        if not entries:
            return

        placeholders = ','.join(['(?, ?, ?, ?)'] * len(entries))
        flat_keys = [val for entry in entries for val in entry[:4]]

        query = f"""
        SELECT removed_id, snapshot_id, model_name, modality
        FROM {self.config.CACHE_TABLE}
        WHERE (removed_id, snapshot_id, model_name, modality) IN ({placeholders})
        """
        try:
            self.cursor.execute("PRAGMA temp_store = MEMORY;")
            self.cursor.execute(query, flat_keys)
            existing = set(self.cursor.fetchall())
        except sqlite3.OperationalError:
            existing = set()

        to_insert = [
            entry for entry in entries
            if tuple(entry[:4]) not in existing
        ]

        if to_insert:
            self.cursor.executemany(f"""
            INSERT INTO {self.config.CACHE_TABLE}
            (removed_id, snapshot_id, model_name, modality, score)
            VALUES (?, ?, ?, ?, ?)
            """, to_insert)
            self.conn.commit()

    def get_top_k(self, removed_id, model_name, modality, k=10, descending=True):
        """
        Retrieve top-k snapshot IDs with highest/lowest similarity scores for a given removed ID.

        Args:
            removed_id (str): Removed method/function ID.
            model_name (str): Model name used for scoring.
            modality (str): Modality used.
            k (int): Number of results to return.
            descending (bool): Sort by descending scores if True.

        Returns:
            list of tuple: List of (removed_id, snapshot_id, score)
        """
        order = "DESC" if descending else "ASC"
        self.cursor.execute(f"""
        SELECT removed_id, snapshot_id, score
        FROM {self.config.CACHE_TABLE}
        WHERE removed_id = ? AND model_name = ? AND modality = ?
        ORDER BY score {order}
        LIMIT ?
        """, (removed_id, model_name, modality, k))
        return self.cursor.fetchall()

    def get_cached_score(self, removed_id, snapshot_id, model_name, modality):
        """
        Retrieves the cached similarity score for a specific comparison.

        Args:
            removed_id (str): Removed method/function ID.
            snapshot_id (str): Matching snapshot method/function ID.
            model_name (str): Model name used for scoring.
            modality (str): Type of comparison (e.g., "text", "ast").

        Returns:
            float or None: Cached score if present, else None.
        """
        self.cursor.execute(f"""
            SELECT score FROM {self.config.CACHE_TABLE}
            WHERE removed_id = ? AND snapshot_id = ? AND model_name = ? AND modality = ?
        """, (removed_id, snapshot_id, model_name, modality))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()
