"""
Migration: add validllm and validmodel columns to expression if missing.
"""
from mwi import model


def _add_column_sql(table: str, coldef: str):
    with model.DB.atomic():
        try:
            model.DB.execute_sql(f'ALTER TABLE {table} ADD COLUMN {coldef}')
            print(f"Added column on {table}: {coldef}")
        except Exception as e:
            msg = str(e).lower()
            if 'duplicate column name' in msg or 'already exists' in msg:
                print(f"Column already exists for {table}: {coldef} — skipping")
            else:
                raise


def upgrade():
    print("Ensuring expression has columns: validllm (TEXT), validmodel (TEXT)…")
    _add_column_sql('expression', 'validllm TEXT DEFAULT NULL')
    _add_column_sql('expression', 'validmodel TEXT DEFAULT NULL')

