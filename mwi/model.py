"""
Core Model definition
"""

from os import path
import settings
import datetime
import json
from peewee import (
    SqliteDatabase,
    Model,
    CharField,
    TextField,
    DateTimeField,
    ForeignKeyField,
    IntegerField,
    CompositeKey,
    BooleanField,
    FloatField
)

DB = SqliteDatabase(path.join(settings.data_location, 'mwi.db'), pragmas={
    'journal_mode': 'wal',
    'cache_size': -1 * 512000,
    'foreign_keys': 1,
    'ignore_check_constrains': 0,
    'synchronous': 0
})


class BaseModel(Model):
    """
    BaseModel
    """

    class Meta:
        """
        Meta class to set DB
        """
        database = DB


class Land(BaseModel):
    """
    Land model
    """
    name = CharField(unique=True)
    description = TextField()
    lang = CharField(max_length=100, default='fr')  # Accepts comma-separated list of languages
    created_at = DateTimeField(default=datetime.datetime.now)


class Domain(BaseModel):
    """
    Domain model
    """
    name = CharField(unique=True)
    http_status = CharField(max_length=3, null=True)
    title = TextField(null=True)
    description = TextField(null=True)
    keywords = TextField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)
    fetched_at = DateTimeField(null=True)


class Expression(BaseModel):
    """
    Expression model
    """
    land = ForeignKeyField(Land, backref='expressions', on_delete='CASCADE')
    url = TextField(index=True)
    domain = ForeignKeyField(Domain, backref='expressions')
    http_status = CharField(max_length=3, null=True, index=True)
    lang = CharField(max_length=100, null=True)  # Accepts comma-separated list of languages
    title = CharField(null=True)
    description = TextField(null=True)
    keywords = TextField(null=True)
    readable = TextField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)
    published_at = DateTimeField(null=True)
    fetched_at = DateTimeField(null=True, index=True)
    approved_at = DateTimeField(null=True)
    readable_at = DateTimeField(null=True, index=True)
    relevance = IntegerField(null=True)
    depth = IntegerField(null=True)
    # LLM validation (OpenRouter gate)
    # validllm stores 'oui'/'non' (fr) to reflect a binary verdict
    validllm = CharField(max_length=3, null=True)
    # validmodel stores the OpenRouter model slug used for the verdict
    validmodel = CharField(max_length=100, null=True)


class ExpressionLink(BaseModel):
    """
    ExpressionLink model
    """

    class Meta:
        """
        Meta class to set composite primary key
        """
        primary_key = CompositeKey('source', 'target')
        table_name = 'expressionlink'

    source = ForeignKeyField(Expression, backref='links_to', on_delete='CASCADE')
    target = ForeignKeyField(Expression, backref='linked_by', on_delete='CASCADE')


class Word(BaseModel):
    """
    Word model
    """
    term = CharField(max_length=30)
    lemma = CharField(max_length=30)


class LandDictionary(BaseModel):
    """
    LandDictionary model
    """

    class Meta:
        """
        Meta class to set composite primary key
        """
        primary_key = CompositeKey('land', 'word')
        table_name = 'landdictionary'

    land = ForeignKeyField(Land, backref='words', on_delete='CASCADE')
    word = ForeignKeyField(Word, backref='lands', on_delete='CASCADE')


class Media(BaseModel):
    """
    Media model enrichi avec analyse
    """
    expression = ForeignKeyField(Expression, backref='medias', on_delete='CASCADE')
    url = TextField()
    type = CharField(max_length=30)
    
    # Dimensions et métadonnées
    width = IntegerField(null=True)
    height = IntegerField(null=True)
    file_size = IntegerField(null=True)
    format = CharField(max_length=10, null=True)
    color_mode = CharField(max_length=10, null=True)
    
    # Analyse visuelle
    dominant_colors = TextField(null=True)
    has_transparency = BooleanField(null=True)
    aspect_ratio = FloatField(null=True)
    
    # Métadonnées avancées
    exif_data = TextField(null=True)
    image_hash = CharField(max_length=64, null=True)
    
    # Analyse de contenu
    content_tags = TextField(null=True)
    nsfw_score = FloatField(null=True)
    
    # Traitement
    analyzed_at = DateTimeField(null=True)
    analysis_error = TextField(null=True)
    websafe_colors = TextField(null=True)
    
    class Meta:
        table_name = 'media'
        indexes = (
            (('width', 'height'), False),
            (('file_size',), False),
            (('image_hash',), False),
            (('analyzed_at',), False),
        )
    
    def is_conforming(self, min_width: int = 0, min_height: int = 0, max_file_size: int = 0) -> bool:
        """Vérifie la conformité aux critères"""
        try:
            # Assurer que les valeurs sont des nombres avant la comparaison
            width = self.width if self.width is not None else 0
            height = self.height if self.height is not None else 0
            file_size = self.file_size if self.file_size is not None else 0

            if min_width > 0 and width < min_width:
                return False
            if min_height > 0 and height < min_height:
                return False
            if max_file_size > 0 and file_size > max_file_size:
                return False
        except (ValueError, TypeError):
            # En cas d'erreur de conversion ou de type, considérer non conforme
            return False
        return True

    def get_dominant_colors_list(self):
        """Retourne la liste des couleurs dominantes"""
        if self.dominant_colors and isinstance(self.dominant_colors, str):
            try:
                return json.loads(self.dominant_colors)
            except json.JSONDecodeError:
                return []
        return []

    def get_exif_dict(self):
        """Retourne le dictionnaire EXIF"""
        if self.exif_data and isinstance(self.exif_data, str):
            try:
                return json.loads(self.exif_data)
            except json.JSONDecodeError:
                return {}
        return {}

    def get_content_tags_list(self):
        """Retourne la liste des tags de contenu"""
        if self.content_tags and isinstance(self.content_tags, str):
            try:
                return json.loads(self.content_tags)
            except json.JSONDecodeError:
                return []
        return []


"""
Client Model Definition
"""


class Paragraph(BaseModel):
    """Représente un paragraphe extrait d'une Expression."""
    expression = ForeignKeyField(Expression, backref='paragraphs', on_delete='CASCADE', index=True)
    domain = ForeignKeyField(Domain, backref='paragraphs', on_delete='CASCADE', index=True)
    para_index = IntegerField(index=True, help_text="Ordre du paragraphe dans l'expression")
    text = TextField()
    text_hash = CharField(max_length=64, unique=True, index=True, help_text="SHA-256 du texte pour déduplication")
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = 'paragraph'


class ParagraphEmbedding(BaseModel):
    """Stocke le vecteur d'embedding pour un paragraphe."""
    paragraph = ForeignKeyField(Paragraph, backref='embedding', unique=True, on_delete='CASCADE')
    embedding = TextField(help_text="Vecteur d'embedding stocké en tant que JSON string")
    norm = FloatField(null=True, help_text="Norme L2 du vecteur pour calcul de similarité optimisé")
    model_name = CharField(max_length=100)
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = 'paragraph_embedding'


class ParagraphSimilarity(BaseModel):
    """Stocke une paire de paragraphes similaires avec leur score."""
    source_paragraph = ForeignKeyField(Paragraph, backref='similar_to', on_delete='CASCADE')
    target_paragraph = ForeignKeyField(Paragraph, backref='similar_from', on_delete='CASCADE')
    score = FloatField(index=True)
    score_raw = FloatField(null=True)
    method = CharField(max_length=50, default='cosine', help_text="Ex: 'cosine', 'cosine+rerank'")
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = 'paragraph_similarity'
        primary_key = CompositeKey('source_paragraph', 'target_paragraph', 'method')
        indexes = (
            (('source_paragraph', 'score'), False),
        )


class Tag(BaseModel):
    """
    Project tag model
    color: hex value string as #FF0022
    """
    land = ForeignKeyField(Land, backref='tags', on_delete='CASCADE')
    parent = ForeignKeyField('self', null=True, backref='children', on_delete='CASCADE')
    name = TextField()
    sorting = IntegerField()
    color = CharField(max_length=7)


class TaggedContent(BaseModel):
    tag = ForeignKeyField(Tag, backref='contents', on_delete='CASCADE')
    expression = ForeignKeyField(Expression, backref='tagged_contents', on_delete='CASCADE')
    text = TextField()
    from_char = IntegerField()
    to_char = IntegerField()
