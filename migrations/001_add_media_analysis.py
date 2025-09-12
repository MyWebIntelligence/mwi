"""
Migration pour ajouter les champs d'analyse média
"""

import datetime
import os
import sys
import importlib.util
from mwi import model

def upgrade():
    """
    Ajoute les colonnes d'analyse média à la table Media
    """
    print("Starting media analysis migration...")
    
    with model.DB.atomic():
        # Vérifier si les colonnes existent déjà
        cursor = model.DB.execute_sql("PRAGMA table_info(media)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        columns_to_add = [
            ('width', 'INTEGER'),
            ('height', 'INTEGER'),
            ('file_size', 'INTEGER'),
            ('format', 'VARCHAR(10)'),
            ('color_mode', 'VARCHAR(10)'),
            ('dominant_colors', 'TEXT'),
            ('has_transparency', 'BOOLEAN'),
            ('aspect_ratio', 'REAL'),
            ('exif_data', 'TEXT'),
            ('image_hash', 'VARCHAR(64)'),
            ('analyzed_at', 'DATETIME'),
            ('analysis_error', 'TEXT')
        ]
        
        # Ajouter seulement les colonnes manquantes
        for column_name, column_type in columns_to_add:
            if column_name not in existing_columns:
                print(f"Adding column: {column_name}")
                model.DB.execute_sql(f'''
                    ALTER TABLE media 
                    ADD COLUMN {column_name} {column_type} DEFAULT NULL
                ''')
            else:
                print(f"Column {column_name} already exists, skipping...")
        
        # Index pour optimiser les requêtes
        print("Creating indexes...")
        indexes_to_create = [
            "CREATE INDEX IF NOT EXISTS idx_media_size ON media(file_size)",
            "CREATE INDEX IF NOT EXISTS idx_media_dimensions ON media(width, height)",
            "CREATE INDEX IF NOT EXISTS idx_media_hash ON media(image_hash)",
            "CREATE INDEX IF NOT EXISTS idx_media_analyzed ON media(analyzed_at)"
        ]
        
        for index_sql in indexes_to_create:
            model.DB.execute_sql(index_sql)
        
    print("Media analysis migration completed successfully")

def downgrade():
    """
    Supprime les colonnes d'analyse média
    """
    print("Reverting media analysis migration...")
    
    # SQLite ne supporte pas DROP COLUMN directement
    # Il faut recréer la table
    with model.DB.atomic():
        model.DB.execute_sql('''
            CREATE TABLE media_backup AS 
            SELECT id, expression_id, url, type 
            FROM media
        ''')
        model.DB.execute_sql('DROP TABLE media')
        model.DB.execute_sql('''
            CREATE TABLE media AS 
            SELECT * FROM media_backup
        ''')
        model.DB.execute_sql('DROP TABLE media_backup')
    
    print("Media analysis migration reverted")


class MigrationManager:
    def __init__(self):
        self.migrations_dir = 'migrations'
        self.migration_table = 'schema_migrations'
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Crée la table des migrations si elle n'existe pas"""
        model.DB.execute_sql('''
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                executed_at DATETIME NOT NULL
            )
        ''')
    
    def get_executed_migrations(self):
        """Retourne la liste des migrations déjà exécutées"""
        cursor = model.DB.execute_sql(
            'SELECT version FROM schema_migrations ORDER BY version'
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_pending_migrations(self):
        """Retourne la liste des migrations à exécuter"""
        executed = set(self.get_executed_migrations())
        
        # Liste tous les fichiers de migration
        migration_files = []
        if os.path.exists(self.migrations_dir):
            for filename in sorted(os.listdir(self.migrations_dir)):
                if filename.endswith('.py') and not filename.startswith('__'):
                    version = filename[:-3]  # Enlève .py
                    if version not in executed:
                        migration_files.append(filename)
        
        return migration_files
    
    def run_migration(self, filename):
        """Exécute une migration"""
        version = filename[:-3]
        filepath = os.path.join(self.migrations_dir, filename)
        
        # Charge le module de migration
        spec = importlib.util.spec_from_file_location(version, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load migration module: {filepath}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Exécute la migration
        print(f"Running migration: {version}")
        module.upgrade()
        
        # Enregistre la migration
        model.DB.execute_sql(
            'INSERT INTO schema_migrations (version, executed_at) VALUES (?, ?)',
            (version, datetime.datetime.now())
        )
        
        print(f"Migration {version} completed")
    
    def migrate(self):
        """Exécute toutes les migrations en attente"""
        pending = self.get_pending_migrations()
        
        if not pending:
            print("No pending migrations")
            return
        
        print(f"Found {len(pending)} pending migrations")
        
        for migration in pending:
            try:
                with model.DB.atomic():
                    self.run_migration(migration)
            except Exception as e:
                print(f"Error running migration {migration}: {e}")
                sys.exit(1)
        
        print("All migrations completed successfully")

if __name__ == '__main__':
    manager = MigrationManager()
    manager.migrate()
