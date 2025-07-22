import os
import pandas as pd
import shutil
from datetime import datetime
from typing import Optional, List, Dict
import glob
import logging

logger = logging.getLogger("DataManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DataManager:
    def __init__(self, data_dir: str = 'data/managed'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _city_file(self, city: str, ext: str = 'parquet', version: Optional[str] = None) -> str:
        city_key = city.lower().replace(' ', '_')
        if version:
            return os.path.join(self.data_dir, f"{city_key}_{version}.{ext}")
        else:
            return os.path.join(self.data_dir, f"{city_key}.{ext}")

    def save(self, df: pd.DataFrame, city: str, ext: str = 'parquet', versioned: bool = True) -> str:
        """Save DataFrame for a city, optionally versioned by timestamp."""
        version = datetime.now().strftime('%Y%m%d_%H%M%S') if versioned else None
        path = self._city_file(city, ext, version)
        if ext == 'csv':
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
        logger.info(f"Saved {len(df)} records for {city} to {path}")
        return path

    def load(self, city: str, ext: str = 'parquet', version: Optional[str] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load DataFrame for a city, optionally by version and columns."""
        path = self._city_file(city, ext, version)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No data file found for {city} at {path}")
        if ext == 'csv':
            df = pd.read_csv(path, usecols=columns)  # type: ignore
        else:
            df = pd.read_parquet(path, columns=columns)
        logger.info(f"Loaded {len(df)} records for {city} from {path}")
        return df

    def list_versions(self, city: str, ext: str = 'parquet') -> List[str]:
        """List all versioned files for a city."""
        city_key = city.lower().replace(' ', '_')
        pattern = os.path.join(self.data_dir, f"{city_key}_*.{ext}")
        files = glob.glob(pattern)
        return sorted(files)

    def backup(self, city: str, ext: str = 'parquet') -> str:
        """
        Create a backup copy of the most recent versioned file for a city.
        - The backup file is named {city}_backup.parquet (or .csv).
        - An archive file is also created: {city}_archive_{timestamp}.parquet, preserving a historical snapshot.
        Returns the backup file path.
        """
        versions = self.list_versions(city, ext)
        if not versions:
            raise FileNotFoundError(f"No versioned file to backup for {city}")
        latest_version = sorted(versions)[-1]
        city_key = city.lower().replace(' ', '_')
        backup_path = os.path.join(self.data_dir, f"{city_key}_backup.{ext}")
        # Copy to backup
        shutil.copy2(latest_version, backup_path)
        # Also create an archive snapshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = os.path.join(self.data_dir, f"{city_key}_archive_{timestamp}.{ext}")
        shutil.copy2(latest_version, archive_path)
        logger.info(f"Backup created: {backup_path}")
        logger.info(f"Archive snapshot created: {archive_path}")
        return backup_path

    def archive_old_versions(self, city: str, keep_last: int = 3, ext: str = 'parquet'):
        """Archive (move) all but the last N versions for a city to an 'archive' subfolder."""
        versions = self.list_versions(city, ext)
        if len(versions) <= keep_last:
            logger.info(f"Nothing to archive for {city}")
            return
        archive_dir = os.path.join(self.data_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        to_archive = versions[:-keep_last]
        for f in to_archive:
            shutil.move(f, os.path.join(archive_dir, os.path.basename(f)))
            logger.info(f"Archived {f}")

    def cleanup_archives(self, days: int = 30):
        """Delete archived files older than N days."""
        archive_dir = os.path.join(self.data_dir, 'archive')
        if not os.path.exists(archive_dir):
            return
        now = datetime.now().timestamp()
        for f in glob.glob(os.path.join(archive_dir, '*')):
            if os.path.isfile(f):
                mtime = os.path.getmtime(f)
                if (now - mtime) > days * 86400:
                    os.remove(f)
                    logger.info(f"Deleted old archive: {f}")

    def load_all_cities(self, ext: str = 'parquet', columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load all city data files into a dict of DataFrames."""
        files = glob.glob(os.path.join(self.data_dir, f"*.{ext}"))
        result = {}
        for f in files:
            city = os.path.basename(f).split('.')[0].split('_')[0]
            if ext == 'parquet':
                result[city] = pd.read_parquet(f, columns=columns)
            else:
                result[city] = pd.read_csv(f, usecols=columns)  # type: ignore
        logger.info(f"Loaded data for {len(result)} cities")
        return result

    def process_in_chunks(self, city: str, func, chunk_size: int = 100_000, ext: str = 'parquet'):
        """Apply a function to each chunk of a large city file (memory efficient)."""
        path = self._city_file(city, ext)
        if ext == 'csv':
            for chunk in pd.read_csv(path, chunksize=chunk_size):
                func(chunk)
        else:
            # Parquet doesn't support chunked reading natively, so read all (or use filters if possible)
            df = pd.read_parquet(path)
            for i in range(0, len(df), chunk_size):
                func(df.iloc[i:i+chunk_size]) 