"""
Migration: add score_raw column to paragraph_similarity if missing.
"""
from mwi import model


def upgrade():
    print("Adding score_raw to paragraph_similarity if not exists…")
    with model.DB.atomic():
        try:
            model.DB.execute_sql('ALTER TABLE paragraph_similarity ADD COLUMN score_raw REAL DEFAULT NULL')
            print("Column 'score_raw' added.")
        except Exception as e:
            msg = str(e).lower()
            if 'duplicate column name' in msg or 'already exists' in msg:
                print("Column 'score_raw' already exists, skipping.")
            else:
                raise

