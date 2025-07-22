import sqlite3
import logging
import pandas as pd
import os

class DatabaseManager:
    def __init__(self, db_path='data/aqi_data.db'):
        self.db_path = db_path
        self.logger = logging.getLogger("DatabaseManager")
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS aqi_data (
                city TEXT,
                state TEXT,
                country TEXT,
                timestamp TEXT,
                source TEXT,
                aqi INTEGER,
                pm25 REAL,
                pm10 REAL,
                no2 REAL,
                so2 REAL,
                co REAL,
                o3 REAL,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                wind_direction REAL,
                visibility REAL,
                cloud_cover REAL,
                precipitation REAL,
                latitude REAL,
                longitude REAL,
                created_at TEXT,
                PRIMARY KEY(city, timestamp)
            )
        """)
        self.conn.commit()

    def insert_aqi_data(self, aqi_data: dict, allow_upsert: bool = False) -> bool:
        try:
            columns = ', '.join(aqi_data.keys())
            placeholders = ', '.join(['?'] * len(aqi_data))
            values = list(aqi_data.values())

            if allow_upsert:
                query = f"""
                    INSERT OR REPLACE INTO aqi_data ({columns})
                    VALUES ({placeholders})
                """
            else:
                query = f"""
                    INSERT INTO aqi_data ({columns})
                    VALUES ({placeholders})
                """

            self.cursor.execute(query, values)
            self.conn.commit()
            return True

        except sqlite3.IntegrityError:
            self.logger.warning(f"⚠️ Duplicate entry skipped for {aqi_data['city']} at {aqi_data['timestamp']}")
            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to insert AQI data: {e}")
            return False

    def export_all_aqi_data_to_csv(self, csv_path: str):
        """
        Merge current in-memory SQLite data with historical CSV (if exists),
        deduplicate on (city, timestamp), and save merged CSV.
        """
        try:
            # Load historical data if exists
            if os.path.exists(csv_path):
                historical_df = pd.read_csv(csv_path)
                self.logger.info(f"📁 Loaded {len(historical_df)} historical rows from {csv_path}")
            else:
                historical_df = pd.DataFrame()

            # Load current in-session data from SQLite
            current_df = pd.read_sql_query("SELECT * FROM aqi_data", self.conn)
            self.logger.info(f"🆕 Loaded {len(current_df)} new rows from database")

            # Combine and deduplicate
            combined_df = pd.concat([historical_df, current_df], ignore_index=True)
            combined_df.drop_duplicates(subset=["city", "timestamp"], keep="last", inplace=True)

            # Export merged data to CSV
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            combined_df.to_csv(csv_path, index=False)
            self.logger.info(f"✅ Exported merged AQI data to {csv_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to export AQI data to CSV: {e}")

    def get_latest_aqi_data(self, city: str) -> dict:
        """Return the latest AQI data for Karachi only."""
        query = "SELECT * FROM aqi_data WHERE city=? ORDER BY timestamp DESC LIMIT 1"
        self.cursor.execute(query, (city,))
        row = self.cursor.fetchone()
        if row:
            columns = [desc[0] for desc in self.cursor.description]
            return dict(zip(columns, row))
        return None

    def get_city_statistics(self, city: str, days: int = 30) -> dict:
        """Return statistics for Karachi only."""
        query = "SELECT * FROM aqi_data WHERE city=? AND timestamp >= date('now', ? )"
        self.cursor.execute(query, (city, f'-{days} days'))
        rows = self.cursor.fetchall()
        if not rows:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        stats = {
            'total_records': len(df),
            'avg_aqi': df['aqi'].mean(),
            'min_aqi': df['aqi'].min(),
            'max_aqi': df['aqi'].max(),
            'avg_temperature': df['temperature'].mean(),
            'avg_humidity': df['humidity'].mean(),
            'avg_pm25': df['pm25'].mean(),
            'avg_pm10': df['pm10'].mean()
        }
        return stats

    def get_database_stats(self):
        query = "SELECT COUNT(*) FROM aqi_data"
        self.cursor.execute(query)
        total_aqi_records = self.cursor.fetchone()[0]
        db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
        return {'total_aqi_records': total_aqi_records, 'database_size_mb': db_size_mb}

    def cleanup_old_data(self, days: int = 90) -> int:
        query = "DELETE FROM aqi_data WHERE timestamp < date('now', ? )"
        self.cursor.execute(query, (f'-{days} days',))
        deleted = self.cursor.rowcount
        self.conn.commit()
        return deleted

    def close(self):
        self.conn.close()


# Exported singleton
_db_manager = DatabaseManager()

def get_db_manager():
    return _db_manager
