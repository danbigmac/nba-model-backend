import threading
import sqlite3, pickle, json, time
import pandas as pd
from contextlib import contextmanager

DB_FILE = "cache/nba_cache.db"

class CacheManager:
    def __init__(self, db_file=DB_FILE):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_tables()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            full_name TEXT,
            first_name TEXT,
            last_name TEXT,
            is_active INTEGER,
            raw_json TEXT,
            last_updated TIMESTAMP
        );
                          
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT,
            created_at TIMESTAMP,
            completed_at TIMESTAMP,
            result_json TEXT
        );
                          
        CREATE TABLE IF NOT EXISTS top_scorers (
            season TEXT,
            player_id INTEGER,
            rank INTEGER,
            pts REAL,
            last_updated TEXT,
            PRIMARY KEY (season, player_id)
        );

        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            full_name TEXT,
            abbreviation TEXT,
            nickname TEXT,
            city TEXT,
            state TEXT,
            year_founded INTEGER,
            raw_json TEXT,
            last_updated TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS player_logs (
            player_id INTEGER,
            season TEXT,
            data BLOB,
            last_updated TIMESTAMP,
            PRIMARY KEY (player_id, season)
        );

        CREATE TABLE IF NOT EXISTS team_stats (
            season TEXT PRIMARY KEY,
            data BLOB,
            last_updated TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS models (
            key TEXT PRIMARY KEY,
            model_blob BLOB
        );
        """)
        self.conn.commit()

    # ---------- PLAYERS ----------
    def save_players(self, players_list):
        with self.lock:
            cur = self.conn.cursor()
            for p in players_list:
                cur.execute("""
                    INSERT OR REPLACE INTO players (player_id, full_name, first_name, last_name, is_active, raw_json, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (p["id"], p["full_name"], p["first_name"], p["last_name"], int(p["is_active"]), json.dumps(p), int(time.time())))
            self.conn.commit()

    def load_players(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT raw_json, last_updated FROM players")
            rows = cur.fetchall()
            if not rows:
                return None, None
            players = [json.loads(r[0]) for r in rows]
            # All players have same last_updated timestamp
            last_updated = rows[0][1]
            return players, last_updated
    
    # ---------- TOP SCORERS ----------
    def save_top_scorers(self, season: str, df):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM top_scorers WHERE season = ?", (season,))
            ts = int(time.time())  # Unix timestamp
            for _, row in df.iterrows():
                cur.execute(
                    "INSERT OR REPLACE INTO top_scorers (season, player_id, rank, pts, last_updated) VALUES (?,?,?,?,?)",
                    (season, int(row["PLAYER_ID"]), int(row["RANK"]), float(row["PTS"]), ts)
                )
            self.conn.commit()

    def load_top_scorers(self, season: str):
        with self.lock:
            df = pd.read_sql_query("SELECT * FROM top_scorers WHERE season = ?", self.conn, params=(season,))
            if df is not None and not df.empty:
                return df, int(df.iloc[0]["last_updated"])
            return None, None

    # ---------- PLAYER LOGS ----------
    def save_player_logs(self, player_id, season, df):
        with self.lock:
            blob = pickle.dumps(df)
            ts = int(time.time())  # Unix timestamp
            cur = self.conn.cursor()
            cur.execute("INSERT OR REPLACE INTO player_logs VALUES (?, ?, ?, ?)", (player_id, season, blob, ts))
            self.conn.commit()

    def load_player_logs(self, player_id, season):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT data, last_updated FROM player_logs WHERE player_id=? AND season=?", (player_id, season))
            row = cur.fetchone()
            if row:
                df = pickle.loads(row[0])
                last_updated = row[1]
                return df, last_updated
            return None, None

    # ---------- TEAM STATS ----------
    def save_team_stats(self, season, df):
        with self.lock:
            blob = pickle.dumps(df)
            ts = int(time.time())  # Unix timestamp
            cur = self.conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO team_stats (season, data, last_updated)
                VALUES (?, ?, ?)
            """, (season, blob, ts))
            self.conn.commit()

    def load_team_stats(self, season):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT data, last_updated FROM team_stats WHERE season=?", (season,))
            row = cur.fetchone()
            if row:
                df = pickle.loads(row[0])
                last_updated = row[1]
                return df, last_updated
            return None, None

    # ---------- TEAMS ----------
    def save_teams(self, teams_list):
        with self.lock:
            cur = self.conn.cursor()
            ts = int(time.time())
            for t in teams_list:
                cur.execute("""
                    INSERT OR REPLACE INTO teams 
                    (team_id, full_name, abbreviation, nickname, city, state, year_founded, raw_json, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t["id"],
                    t["full_name"],
                    t["abbreviation"],
                    t["nickname"],
                    t["city"],
                    t.get("state", ""),   # not always present
                    t.get("year_founded", None),
                    json.dumps(t),
                    ts
                ))
            self.conn.commit()

    def load_teams(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT raw_json, last_updated FROM teams")
            rows = cur.fetchall()
            if not rows:
                return None, None
            return [json.loads(r[0]) for r in rows], rows[0][1]

    # ---------- MODELS ----------
    def save_model(self, key, model):
        with self.lock:
            blob = pickle.dumps(model)
            cur = self.conn.cursor()
            cur.execute("INSERT OR REPLACE INTO models VALUES (?, ?)", (key, blob))
            self.conn.commit()

    def load_model(self, key):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT model_blob FROM models WHERE key=?", (key,))
            row = cur.fetchone()
            if row:
                return pickle.loads(row[0])
            return None

    def execute_query(self, query: str, params: tuple = ()):
        """General-purpose query with lock."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(query, params)
            self.conn.commit()
            return cur.fetchall()