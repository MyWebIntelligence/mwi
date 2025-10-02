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
    """Base model class for all database models.

    This class extends Peewee's Model class and provides the base database
    configuration for all models in the application. All models should inherit
    from this class to ensure they use the correct database connection.

    Attributes:
        Meta: Inner class defining database connection settings.

    Notes:
        This base model is configured to use the SQLite database with WAL mode,
        optimized cache settings, and foreign key constraints enabled.
    """

    class Meta:
        """Meta configuration class for BaseModel.

        Attributes:
            database: The SQLite database instance to use for all models.
        """
        database = DB


class Land(BaseModel):
    """Research project or thematic collection model.

    A Land represents a research project or thematic collection that groups
    related web expressions. It serves as the primary organizational unit for
    web intelligence research activities.

    Attributes:
        name (CharField): Unique identifier name for the land.
        description (TextField): Detailed description of the research topic.
        lang (CharField): Language code(s) for the land. Accepts comma-separated
            list of language codes. Defaults to 'fr' (French).
        created_at (DateTimeField): Timestamp when the land was created.
            Automatically set to current datetime on creation.

    Notes:
        The name field must be unique across all lands in the database.
        Multiple languages can be specified as a comma-separated list
        (e.g., 'fr,en,de').
    """
    name = CharField(unique=True)
    description = TextField()
    lang = CharField(max_length=100, default='fr')  # Accepts comma-separated list of languages
    created_at = DateTimeField(default=datetime.datetime.now)


class Domain(BaseModel):
    """Website domain model with metadata.

    Represents a unique website or domain that has been crawled. Stores
    domain-level metadata extracted from the homepage or root URL.

    Attributes:
        name (CharField): Unique domain name (e.g., 'example.com').
        http_status (CharField): HTTP status code from the last fetch attempt.
            Maximum 3 characters, nullable.
        title (TextField): Page title from the domain's homepage, nullable.
        description (TextField): Meta description from the domain's homepage,
            nullable.
        keywords (TextField): Meta keywords from the domain's homepage, nullable.
        created_at (DateTimeField): Timestamp when the domain was first added.
            Automatically set to current datetime on creation.
        fetched_at (DateTimeField): Timestamp of the last successful fetch,
            nullable.

    Notes:
        The name field must be unique across all domains.
        HTTP status codes are stored as strings to preserve leading zeros.
    """
    name = CharField(unique=True)
    http_status = CharField(max_length=3, null=True)
    title = TextField(null=True)
    description = TextField(null=True)
    keywords = TextField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)
    fetched_at = DateTimeField(null=True)


class Expression(BaseModel):
    """Individual web page or URL within a research land.

    Represents a single web page that has been crawled and analyzed. Contains
    metadata, content, relevance scoring, and optional LLM validation results.

    Attributes:
        land (ForeignKeyField): Reference to the parent Land. Cascade deletes
            when the land is deleted.
        url (TextField): The full URL of the expression. Indexed for performance.
        domain (ForeignKeyField): Reference to the Domain this expression belongs to.
        http_status (CharField): HTTP status code from fetch attempt. Maximum
            3 characters, indexed, nullable.
        lang (CharField): Language code(s) of the content. Accepts comma-separated
            list of language codes, nullable.
        title (CharField): Page title, nullable.
        description (TextField): Meta description or summary, nullable.
        keywords (TextField): Meta keywords or extracted key terms, nullable.
        readable (TextField): Cleaned readable content extracted by Mercury Parser
            or similar tools, nullable.
        created_at (DateTimeField): Timestamp when the expression was created.
            Automatically set to current datetime on creation.
        published_at (DateTimeField): Publication date of the content, nullable.
        fetched_at (DateTimeField): Timestamp of the last fetch. Indexed, nullable.
        approved_at (DateTimeField): Timestamp when the expression was approved
            for inclusion in research, nullable.
        readable_at (DateTimeField): Timestamp when readable content was extracted.
            Indexed, nullable.
        relevance (IntegerField): Calculated relevance score based on keyword
            matching and other heuristics, nullable.
        depth (IntegerField): Crawl depth from the seed URL, nullable.
        validllm (CharField): LLM validation verdict as 'oui' or 'non' (French).
            Maximum 3 characters, nullable.
        validmodel (CharField): OpenRouter model slug used for LLM validation.
            Maximum 100 characters, nullable.
        seorank (TextField): Raw JSON payload from SEO Rank API for this URL,
            nullable.

    Notes:
        The url field is indexed for efficient lookup.
        Multiple languages can be specified as comma-separated values.
        LLM validation fields store binary verdicts and model information.
        The readable field stores cleaned content suitable for analysis.
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
    # seorank persists the raw JSON payload returned by the SEO Rank API for the URL
    seorank = TextField(null=True)


class ExpressionLink(BaseModel):
    """Directed link relationship between two expressions.

    Represents a hyperlink from one expression (source) to another (target).
    Used to construct the link graph for network analysis and relationship mapping.

    Attributes:
        source (ForeignKeyField): The expression containing the link. Cascade
            deletes when the source expression is deleted.
        target (ForeignKeyField): The expression being linked to. Cascade deletes
            when the target expression is deleted.
        Meta: Inner class defining composite primary key and table name.

    Notes:
        The composite primary key ensures each directed link pair is unique.
        Links are directional: source -> target.
        Backref 'links_to' on source allows accessing outgoing links.
        Backref 'linked_by' on target allows accessing incoming links.
    """

    class Meta:
        """Meta configuration for ExpressionLink model.

        Attributes:
            primary_key (CompositeKey): Composite key of source and target fields.
            table_name (str): Custom table name 'expressionlink'.
        """
        primary_key = CompositeKey('source', 'target')
        table_name = 'expressionlink'

    source = ForeignKeyField(Expression, backref='links_to', on_delete='CASCADE')
    target = ForeignKeyField(Expression, backref='linked_by', on_delete='CASCADE')


class Word(BaseModel):
    """Normalized vocabulary word with lemmatization.

    Stores words with their normalized lemma form for keyword matching and
    relevance scoring. Used in conjunction with LandDictionary for land-specific
    terminology.

    Attributes:
        term (CharField): The original word form. Maximum 30 characters.
        lemma (CharField): The normalized/lemmatized form of the word.
            Maximum 30 characters.

    Notes:
        Lemmatization is typically performed using NLTK or similar tools.
        Words are normalized to improve keyword matching accuracy.
        The same lemma may have multiple term variations.
    """
    term = CharField(max_length=30)
    lemma = CharField(max_length=30)


class LandDictionary(BaseModel):
    """Many-to-many relationship between lands and words.

    Associates specific words with research lands to create land-specific
    dictionaries. Used for relevance scoring and keyword matching within
    a particular research context.

    Attributes:
        land (ForeignKeyField): Reference to the Land. Cascade deletes when
            the land is deleted.
        word (ForeignKeyField): Reference to the Word. Cascade deletes when
            the word is deleted.
        Meta: Inner class defining composite primary key and table name.

    Notes:
        The composite primary key ensures each land-word pair is unique.
        Backref 'words' on land allows accessing all words in the dictionary.
        Backref 'lands' on word allows finding all lands using that word.
    """

    class Meta:
        """Meta configuration for LandDictionary model.

        Attributes:
            primary_key (CompositeKey): Composite key of land and word fields.
            table_name (str): Custom table name 'landdictionary'.
        """
        primary_key = CompositeKey('land', 'word')
        table_name = 'landdictionary'

    land = ForeignKeyField(Land, backref='words', on_delete='CASCADE')
    word = ForeignKeyField(Word, backref='lands', on_delete='CASCADE')


class Media(BaseModel):
    """Media content model with comprehensive analysis metadata.

    Represents media files (images, videos, audio) extracted from expressions.
    Includes detailed analysis metadata such as dimensions, colors, EXIF data,
    and perceptual hashing for duplicate detection.

    Attributes:
        expression (ForeignKeyField): Reference to the parent Expression.
            Cascade deletes when the expression is deleted.
        url (TextField): URL of the media file.
        type (CharField): Media type (e.g., 'image', 'video', 'audio').
            Maximum 30 characters.
        width (IntegerField): Media width in pixels, nullable.
        height (IntegerField): Media height in pixels, nullable.
        file_size (IntegerField): File size in bytes, nullable.
        format (CharField): File format (e.g., 'JPEG', 'PNG', 'MP4').
            Maximum 10 characters, nullable.
        color_mode (CharField): Color mode (e.g., 'RGB', 'RGBA', 'L').
            Maximum 10 characters, nullable.
        dominant_colors (TextField): JSON string of dominant colors extracted
            from the media, nullable.
        has_transparency (BooleanField): Whether the media has transparency,
            nullable.
        aspect_ratio (FloatField): Width/height ratio, nullable.
        exif_data (TextField): JSON string of EXIF metadata, nullable.
        image_hash (CharField): Perceptual hash for duplicate detection.
            Maximum 64 characters, nullable.
        content_tags (TextField): JSON string of content tags, nullable.
        nsfw_score (FloatField): Not-safe-for-work score if analyzed, nullable.
        analyzed_at (DateTimeField): Timestamp when analysis was performed,
            nullable.
        analysis_error (TextField): Error message if analysis failed, nullable.
        websafe_colors (TextField): JSON string of web-safe colors, nullable.
        Meta: Inner class defining table name and indexes.

    Notes:
        JSON fields (dominant_colors, exif_data, content_tags, websafe_colors)
        should be serialized/deserialized using helper methods.
        The image_hash field uses perceptual hashing for similarity detection.
        Multiple indexes optimize queries on dimensions, size, hash, and analysis date.
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
        """Meta configuration for Media model.

        Attributes:
            table_name (str): Custom table name 'media'.
            indexes (tuple): Composite and single-field indexes for performance.
        """
        table_name = 'media'
        indexes = (
            (('width', 'height'), False),
            (('file_size',), False),
            (('image_hash',), False),
            (('analyzed_at',), False),
        )
    
    def is_conforming(self, min_width: int = 0, min_height: int = 0, max_file_size: int = 0) -> bool:
        """Check if media meets specified dimension and size criteria.

        Args:
            min_width (int): Minimum width in pixels. Defaults to 0 (no limit).
            min_height (int): Minimum height in pixels. Defaults to 0 (no limit).
            max_file_size (int): Maximum file size in bytes. Defaults to 0 (no limit).

        Returns:
            bool: True if media meets all specified criteria, False otherwise.

        Notes:
            Null values are treated as 0 for comparison purposes.
            Returns False if conversion errors occur.
            A limit of 0 means no constraint is applied for that dimension.
        """
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
        """Get the list of dominant colors from the media.

        Returns:
            list: List of dominant colors as dictionaries or tuples.
                Returns empty list if no colors are stored or on JSON decode error.

        Notes:
            The dominant_colors field should be a JSON-serialized list.
            Returns empty list for invalid JSON or non-string values.
        """
        if self.dominant_colors and isinstance(self.dominant_colors, str):
            try:
                return json.loads(self.dominant_colors)
            except json.JSONDecodeError:
                return []
        return []

    def get_exif_dict(self):
        """Get the EXIF metadata dictionary from the media.

        Returns:
            dict: Dictionary containing EXIF metadata fields.
                Returns empty dictionary if no EXIF data is stored or on JSON decode error.

        Notes:
            The exif_data field should be a JSON-serialized dictionary.
            Returns empty dictionary for invalid JSON or non-string values.
            EXIF data typically includes camera settings, timestamps, GPS coordinates, etc.
        """
        if self.exif_data and isinstance(self.exif_data, str):
            try:
                return json.loads(self.exif_data)
            except json.JSONDecodeError:
                return {}
        return {}

    def get_content_tags_list(self):
        """Get the list of content tags from the media.

        Returns:
            list: List of content tags describing the media.
                Returns empty list if no tags are stored or on JSON decode error.

        Notes:
            The content_tags field should be a JSON-serialized list.
            Returns empty list for invalid JSON or non-string values.
            Content tags may include object detection, scene classification, etc.
        """
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
    """Paragraph extracted from an expression for text analysis.

    Represents a text paragraph extracted from an expression's content.
    Includes deduplication via text hashing and supports semantic similarity
    analysis through embeddings.

    Attributes:
        expression (ForeignKeyField): Reference to the parent Expression.
            Indexed for efficient queries. Cascade deletes when expression is deleted.
        domain (ForeignKeyField): Reference to the Domain this paragraph originates from.
            Indexed for efficient queries. Cascade deletes when domain is deleted.
        para_index (IntegerField): Sequential index of the paragraph within the expression.
            Indexed for ordering.
        text (TextField): The actual paragraph text content.
        text_hash (CharField): SHA-256 hash of the text for deduplication.
            Maximum 64 characters, unique, indexed.
        created_at (DateTimeField): Timestamp when the paragraph was created.
            Automatically set to current datetime on creation.
        Meta: Inner class defining custom table name.

    Notes:
        The text_hash field ensures that duplicate paragraphs are not stored.
        Paragraphs are ordered within an expression using para_index.
        Used in conjunction with ParagraphEmbedding for semantic analysis.
    """
    expression = ForeignKeyField(Expression, backref='paragraphs', on_delete='CASCADE', index=True)
    domain = ForeignKeyField(Domain, backref='paragraphs', on_delete='CASCADE', index=True)
    para_index = IntegerField(index=True, help_text="Ordre du paragraphe dans l'expression")
    text = TextField()
    text_hash = CharField(max_length=64, unique=True, index=True, help_text="SHA-256 du texte pour déduplication")
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        """Meta configuration for Paragraph model.

        Attributes:
            table_name (str): Custom table name 'paragraph'.
        """
        table_name = 'paragraph'


class ParagraphEmbedding(BaseModel):
    """Semantic embedding vector for a paragraph.

    Stores the vector representation of a paragraph for semantic similarity
    analysis. Each paragraph can have one embedding associated with it.

    Attributes:
        paragraph (ForeignKeyField): Reference to the parent Paragraph.
            Unique constraint ensures one embedding per paragraph.
            Cascade deletes when paragraph is deleted.
        embedding (TextField): JSON string representation of the embedding vector.
        norm (FloatField): L2 norm of the embedding vector for optimized
            similarity calculations, nullable.
        model_name (CharField): Name of the embedding model used.
            Maximum 100 characters.
        created_at (DateTimeField): Timestamp when the embedding was created.
            Automatically set to current datetime on creation.
        Meta: Inner class defining custom table name.

    Notes:
        The embedding field stores a serialized vector (typically a list of floats).
        The norm field can be precomputed to optimize cosine similarity calculations.
        Different model_name values indicate embeddings from different models.
    """
    paragraph = ForeignKeyField(Paragraph, backref='embedding', unique=True, on_delete='CASCADE')
    embedding = TextField(help_text="Vecteur d'embedding stocké en tant que JSON string")
    norm = FloatField(null=True, help_text="Norme L2 du vecteur pour calcul de similarité optimisé")
    model_name = CharField(max_length=100)
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        """Meta configuration for ParagraphEmbedding model.

        Attributes:
            table_name (str): Custom table name 'paragraph_embedding'.
        """
        table_name = 'paragraph_embedding'


class ParagraphSimilarity(BaseModel):
    """Similarity relationship between two paragraphs.

    Stores computed similarity scores between pairs of paragraphs using
    various methods (e.g., cosine similarity, reranking). Supports multiple
    similarity calculation methods for the same paragraph pair.

    Attributes:
        source_paragraph (ForeignKeyField): Reference to the source Paragraph.
            Cascade deletes when paragraph is deleted.
        target_paragraph (ForeignKeyField): Reference to the target Paragraph.
            Cascade deletes when paragraph is deleted.
        score (FloatField): Final similarity score. Indexed for efficient
            retrieval of top similar paragraphs.
        score_raw (FloatField): Raw similarity score before any adjustments
            or normalization, nullable.
        method (CharField): Similarity calculation method used.
            Maximum 50 characters. Examples: 'cosine', 'cosine+rerank'.
            Defaults to 'cosine'.
        created_at (DateTimeField): Timestamp when the similarity was computed.
            Automatically set to current datetime on creation.
        Meta: Inner class defining composite primary key, table name, and indexes.

    Notes:
        The composite primary key allows the same paragraph pair to have
        multiple similarity scores using different methods.
        The (source_paragraph, score) index optimizes queries for finding
        the most similar paragraphs to a given source.
        Backref 'similar_to' on source allows finding similar paragraphs.
        Backref 'similar_from' on target allows reverse lookup.
    """
    source_paragraph = ForeignKeyField(Paragraph, backref='similar_to', on_delete='CASCADE')
    target_paragraph = ForeignKeyField(Paragraph, backref='similar_from', on_delete='CASCADE')
    score = FloatField(index=True)
    score_raw = FloatField(null=True)
    method = CharField(max_length=50, default='cosine', help_text="Ex: 'cosine', 'cosine+rerank'")
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        """Meta configuration for ParagraphSimilarity model.

        Attributes:
            table_name (str): Custom table name 'paragraph_similarity'.
            primary_key (CompositeKey): Composite key of source_paragraph,
                target_paragraph, and method fields.
            indexes (tuple): Composite index on (source_paragraph, score)
                for efficient similarity queries.
        """
        table_name = 'paragraph_similarity'
        primary_key = CompositeKey('source_paragraph', 'target_paragraph', 'method')
        indexes = (
            (('source_paragraph', 'score'), False),
        )


class Tag(BaseModel):
    """Hierarchical tag for categorizing content within a land.

    Represents a tag that can be applied to content snippets within expressions.
    Supports hierarchical organization through parent-child relationships.

    Attributes:
        land (ForeignKeyField): Reference to the parent Land this tag belongs to.
            Cascade deletes when the land is deleted.
        parent (ForeignKeyField): Reference to the parent Tag for hierarchical
            organization. Self-referential foreign key. Nullable for root tags.
            Cascade deletes when the parent tag is deleted.
        name (TextField): Display name of the tag.
        sorting (IntegerField): Sort order for displaying tags.
            Lower values appear first.
        color (CharField): Hex color code for the tag (e.g., '#FF0022').
            Maximum 7 characters including the '#' prefix.

    Notes:
        Tags can form a tree structure through the parent relationship.
        Backref 'children' on parent allows accessing child tags.
        The color field should be a valid hex color code starting with '#'.
        Sorting field allows custom ordering independent of alphabetical order.
    """
    land = ForeignKeyField(Land, backref='tags', on_delete='CASCADE')
    parent = ForeignKeyField('self', null=True, backref='children', on_delete='CASCADE')
    name = TextField()
    sorting = IntegerField()
    color = CharField(max_length=7)


class TaggedContent(BaseModel):
    """Content snippet associated with a tag.

    Represents a specific portion of an expression's text that has been tagged.
    Stores the exact text and character positions for precise content marking.

    Attributes:
        tag (ForeignKeyField): Reference to the Tag applied to this content.
            Cascade deletes when the tag is deleted.
        expression (ForeignKeyField): Reference to the Expression containing
            this content. Cascade deletes when the expression is deleted.
        text (TextField): The actual text content that has been tagged.
        from_char (IntegerField): Starting character position in the expression's
            content (0-indexed).
        to_char (IntegerField): Ending character position in the expression's
            content (exclusive, 0-indexed).

    Notes:
        The character positions (from_char, to_char) define a range [from_char, to_char).
        Backref 'contents' on tag allows accessing all content for a tag.
        Backref 'tagged_contents' on expression allows finding all tagged portions.
        The text field provides a denormalized copy for quick access.
    """
    tag = ForeignKeyField(Tag, backref='contents', on_delete='CASCADE')
    expression = ForeignKeyField(Expression, backref='tagged_contents', on_delete='CASCADE')
    text = TextField()
    from_char = IntegerField()
    to_char = IntegerField()
