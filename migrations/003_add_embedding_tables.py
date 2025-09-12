"""
Migration pour ajouter les tables d'embeddings de paragraphes.
Crée: paragraph, paragraph_embedding, paragraph_similarity et leurs index.
Idempotent: utilise CREATE TABLE IF NOT EXISTS et CREATE INDEX IF NOT EXISTS.
"""
from mwi import model


def upgrade():
    print("Starting embedding tables migration…")
    with model.DB.atomic():
        # Paragraph table
        model.DB.execute_sql(
            """
            CREATE TABLE IF NOT EXISTS paragraph (
                id INTEGER PRIMARY KEY,
                expression_id INTEGER NOT NULL REFERENCES expression(id) ON DELETE CASCADE,
                domain_id INTEGER NOT NULL REFERENCES domain(id) ON DELETE CASCADE,
                para_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                text_hash VARCHAR(64) NOT NULL UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        model.DB.execute_sql("CREATE INDEX IF NOT EXISTS idx_paragraph_expr ON paragraph(expression_id)")
        model.DB.execute_sql("CREATE INDEX IF NOT EXISTS idx_paragraph_domain ON paragraph(domain_id)")
        model.DB.execute_sql("CREATE INDEX IF NOT EXISTS idx_paragraph_pidx ON paragraph(para_index)")

        # Paragraph Embedding
        model.DB.execute_sql(
            """
            CREATE TABLE IF NOT EXISTS paragraph_embedding (
                id INTEGER PRIMARY KEY,
                paragraph_id INTEGER NOT NULL UNIQUE REFERENCES paragraph(id) ON DELETE CASCADE,
                embedding TEXT NOT NULL,
                norm REAL,
                model_name VARCHAR(100) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        model.DB.execute_sql("CREATE INDEX IF NOT EXISTS idx_pemb_paragraph ON paragraph_embedding(paragraph_id)")
        model.DB.execute_sql("CREATE INDEX IF NOT EXISTS idx_pemb_model ON paragraph_embedding(model_name)")

        # Paragraph Similarity
        model.DB.execute_sql(
            """
            CREATE TABLE IF NOT EXISTS paragraph_similarity (
                source_paragraph_id INTEGER NOT NULL REFERENCES paragraph(id) ON DELETE CASCADE,
                target_paragraph_id INTEGER NOT NULL REFERENCES paragraph(id) ON DELETE CASCADE,
                score REAL NOT NULL,
                method VARCHAR(50) NOT NULL DEFAULT 'cosine',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_paragraph_id, target_paragraph_id, method)
            )
            """
        )
        model.DB.execute_sql("CREATE INDEX IF NOT EXISTS idx_psim_source_score ON paragraph_similarity(source_paragraph_id, score)")

    print("Embedding tables migration complete.")

