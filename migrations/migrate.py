import os
import importlib.util
from datetime import datetime
from mwi import model

class MigrationManager:
    def __init__(self):
        # Use this file's directory (supports hidden folder name '.migrations')
        self.migrations_dir = os.path.dirname(__file__)
        self.migration_table = 'schema_migrations'
        self._ensure_migration_table()

    def _ensure_migration_table(self):
        model.DB.execute_sql(f'''
            CREATE TABLE IF NOT EXISTS {self.migration_table} (
                version TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
        ''')

    def get_applied_migrations(self):
        cursor = model.DB.execute_sql(f'SELECT version FROM {self.migration_table}')
        return {row[0] for row in cursor.fetchall()}

    def run_pending_migrations(self):
        applied = self.get_applied_migrations()
        
        for filename in sorted(os.listdir(self.migrations_dir)):
            if filename.endswith('.py') and not filename.startswith('__') and filename != 'migrate.py':
                version = os.path.splitext(filename)[0]
                if version not in applied:
                    self.apply_migration(version)

    def apply_migration(self, version):
        module_path = os.path.join(self.migrations_dir, f'{version}.py')
        spec = importlib.util.spec_from_file_location(version, module_path)
        migration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration_module)
        
        print(f"Applying migration {version}...")
        migration_module.upgrade()
        
        model.DB.execute_sql(
            f'INSERT INTO {self.migration_table} (version, applied_at) VALUES (?, ?)',
            (version, datetime.now().isoformat())
        )
        print(f"Migration {version} applied successfully.")
