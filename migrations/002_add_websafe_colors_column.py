"""
Migration pour ajouter la colonne websafe_colors à la table Media.
"""
from mwi import model

def upgrade():
    """
    Ajoute la colonne websafe_colors à la table Media.
    """
    print("Démarrage de la migration pour ajouter websafe_colors...")
    
    with model.DB.atomic():
        try:
            model.DB.execute_sql('ALTER TABLE media ADD COLUMN websafe_colors TEXT DEFAULT NULL')
            print("Colonne 'websafe_colors' ajoutée avec succès.")
        except Exception as e:
            # Ignorer l'erreur si la colonne existe déjà pour rendre la migration ré-exécutable
            if "duplicate column name" in str(e).lower():
                print("La colonne 'websafe_colors' existe déjà. Migration ignorée.")
            else:
                raise e

def downgrade():
    """
    Supprime la colonne websafe_colors.
    NOTE : La suppression de colonnes dans SQLite est complexe et nécessite une recréation de table.
    Cette fonction est fournie à titre indicatif et doit être utilisée avec prudence.
    """
    print("Annulation de la migration websafe_colors...")
    
    with model.DB.atomic():
        # Sauvegarder les données existantes sans la colonne à supprimer
        model.DB.execute_sql('''
            CREATE TABLE media_backup AS 
            SELECT id, expression_id, url, type, width, height, file_size, format, 
                   color_mode, dominant_colors, has_transparency, aspect_ratio, 
                   exif_data, image_hash, content_tags, nsfw_score, 
                   analyzed_at, analysis_error
            FROM media
        ''')
        # Supprimer l'ancienne table
        model.DB.execute_sql('DROP TABLE media')
        # Recréer la table avec l'ancien schéma (sans la nouvelle colonne)
        # Ceci suppose que le modèle Media dans model.py n'a pas encore été mis à jour
        # Il est plus sûr de se baser sur un schéma explicite si possible.
        # Pour cet exemple, nous allons recréer la table à partir du backup.
        model.DB.execute_sql('CREATE TABLE media AS SELECT * FROM media_backup')
        # Supprimer le backup
        model.DB.execute_sql('DROP TABLE media_backup')
    
    print("La colonne 'websafe_colors' a été supprimée.")
