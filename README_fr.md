# My Web Intelligence (MyWI)

Version anglaise : [README.md](README.md)

## Sommaire
- [Vue d’ensemble](#vue-densemble)
- [Fonctionnalités clés](#fonctionnalités-clés)
- [Architecture](#architecture)
  - [Flux de commande](#flux-de-commande)
  - [Modèle de données](#modèle-de-données)
  - [Pipelines & services](#pipelines--services)
  - [Paramètres & configuration](#paramètres--configuration)
- [Mise en route](#mise-en-route)
  - [Docker Compose](#docker-compose)
  - [Docker CLI](#docker-cli)
  - [Développement local](#développement-local)
- [Guide CLI](#guide-cli)
  - [Lands & vocabulaire](#lands--vocabulaire)
  - [Collecte de données](#collecte-de-données)
  - [Contenu lisible](#contenu-lisible)
  - [Médias & domaines](#médias--domaines)
  - [Exports & tags](#exports--tags)
- [Enrichissement & IA](#enrichissement--ia)
  - [SEO Rank](#seo-rank)
  - [Embeddings & similarité](#embeddings--similarité)
  - [Validation LLM en masse](#validation-llm-en-masse)
- [Tests & qualité](#tests--qualité)
- [Dépannage](#dépannage)
- [Licence](#licence)

## Vue d’ensemble

MyWebIntelligence (MyWI) est un outil Python pensé pour les équipes en humanités numériques qui doivent collecter, organiser et analyser de grands volumes de contenus web. Les projets sont regroupés en « lands » qui stockent vocabulaire, URLs, pages crawlées, médias, liens sémantiques et tags qualitatifs dans une base SQLite. Le CLI orchestre crawl, pipelines d’enrichissement et exports pour passer rapidement d’URLs de départ à des jeux de données exploitables.

Toutes les données résident dans SQLite (mode WAL), ce qui facilite l’inspection via des outils comme [DB Browser for SQLite](https://sqlitebrowser.org/) ou l’intégration dans des tableaux de bord.

## Fonctionnalités clés

- Espace de travail centré sur les lands : vocabulaire, URLs, médias et exports isolés par sujet.
- Crawl robuste avec parallélisme poli, filtrage par statut HTTP, profondeur, retries et archivage HTML optionnel.
- Extraction lisible de haute qualité via le pipeline Mercury (stratégies de fusion, enrichissement des métadonnées, reconstruction des liens et recalcul de la pertinence).
- Pipeline d’analyse médias pour images, vidéos et audio (métadonnées, hachages, détection de doublons, couleurs dominantes, score NSFW).
- Intelligence domaine : crawls dédiés, mises à jour heuristiques et contrôleurs fins.
- Surface d’export riche : CSV, GEXF (pages ou nœuds), listings médias, matrices de tags et corpus brut.
- Embeddings sémantiques par paragraphe avec similarité cosine LSH ou NLI et export CSV des pseudolinks.
- Enrichissement SERP & SEO via SerpAPI (seed URLs) et SEO Rank (métriques rafraîchissables).
- Modération LLM de la pertinence (filtre OpenRouter et validation de masse) sans casser les workflows existants.
- Modèle de configuration extensible : timeouts, heuristiques, providers et secrets gérés dans `settings.py` ou par variables d’environnement.

## Architecture

### Flux de commande

```
mywi.py → mwi/cli.py → mwi/controller.py → mwi/core.py / mwi/export.py
                                 ↘ mwi/model.py (Peewee ORM)
                                 ↘ mwi/readable_pipeline.py
                                 ↘ mwi/media_analyzer.py
                                 ↘ mwi/embedding_pipeline.py
```

- `mywi.py` : point d’entrée CLI, charge la configuration et délègue au parseur.
- `mwi/cli.py` : construit l’interface `argparse`, convertit les appels en commandes et expose `command_run()` pour les tests.
- `mwi/controller.py` : fine façade qui mappe les verbes vers la logique `core/export/model` et renvoie `1` en succès (`0` sinon).
- `mwi/core.py` : cœur métier (crawl, parsing, heuristiques, consolidation, raccordement médias).
- `mwi/export.py` : centre des exports CSV, GEXF, médias, tags et corpus.

### Modèle de données

MyWI repose sur SQLite via Peewee. Entités principales :

- `Land` : métadonnées du workspace (nom, langue, description, timestamps).
- `Word` et `LandDictionary` : vocabulaire canonique et liaison many-to-many avec les lands.
- `Domain` : informations par domaine (crawl, métas, http_status).
- `Expression` : URL/page avec profondeur de crawl, contenu lisible, scores, payload SEO, verdict LLM et horodatages.
- `ExpressionLink` : graphe dirigé des hyperliens entre expressions.
- `Media` : ressources images/vidéos/audio détectées, avec métadonnées, hachages et statut d’analyse.
- `Paragraph`, `ParagraphEmbedding`, `ParagraphSimilarity` : texte par paragraphe, vecteurs et liaisons sémantiques (pseudolinks).
- `Tag` et `TaggedContent` : annotations qualitatives hiérarchiques et extraits associés.

### Pipelines & services

- **Pipeline lisible (`mwi/readable_pipeline.py`)** — intègre Mercury Parser avec stratégies (`smart_merge`, `mercury_priority`, `preserve_existing`), reconstruit liens/médias et rafraîchit la pertinence.
- **Analyseur médias (`mwi/media_analyzer.py`)** — traitement asynchrone des médias avec Playwright optionnel pour contenu dynamique ; gestion suppression, statistiques et doublons.
- **Pipeline embeddings (`mwi/embedding_pipeline.py`)** — embeddings de paragraphe via OpenAI, Mistral, Gemini, Hugging Face, provider HTTP ou Ollama, stockage des vecteurs et relations.
- **Heuristiques & enrichissements** — consolidation, mises à jour heuristiques, SEO Rank et autres contrôleurs spécialisés côtoient le cœur applicatif.

### Paramètres & configuration

La configuration runtime est centralisée dans `settings.py` et peut être surchargée via l’environnement.

- **Chemins** — `data_location` (base de données, exports, archives) ; défaut `./data` en local ou `/app/data` en Docker.
- **Réseau** — `user_agent`, `parallel_connections`, `default_timeout`, retries/backoff, archivage HTML optionnel.
- **Pipelines** — valeurs par défaut pour fusion lisible, parallélisme analyse médias, seuils heuristiques et comportement des exports.
- **Clés API**
  - SerpAPI (`settings.serpapi_api_key` / `MWI_SERPAPI_API_KEY`)
  - SEO Rank (`settings.seorank_api_key` / `MWI_SEORANK_API_KEY`)
  - OpenRouter (`settings.openrouter_*` / `MWI_OPENROUTER_*`)
  - Providers d’embeddings (`embed_openai_api_key`, `embed_mistral_api_key`, `embed_gemini_api_key`, `embed_hf_api_key`, etc.)
- **Transport embeddings** — `embed_provider`, `embed_api_url`, `embed_model_name`, taille de lot, bornes de caractères, seuils de similarité.

## Mise en route

Le chemin le plus simple passe par Docker Compose, mais le CLI fonctionne aussi via Docker natif ou un environnement virtuel local. Avant de lancer les pipelines, initialisez la base et pointez `settings.py:data_location` vers un dossier accessible en écriture.

### Docker Compose

Pré-requis : Docker Desktop (ou Docker Engine) avec Compose.

1. Copier le fichier d’exemple et choisir un répertoire de stockage.
   ```bash
   cp .env.example .env
   ```
   Conserver `HOST_DATA_DIR=./data` (défaut) ou définir un chemin absolu (ex. `/Users/vous/mywi_data`).
2. Construire et démarrer les services.
   ```bash
   docker compose up -d --build
   ```
3. Initialiser la base (premier lancement).
   ```bash
   docker compose exec mwi python mywi.py db setup
   ```
4. Utiliser le CLI depuis le conteneur.
   ```bash
   docker compose exec mwi python mywi.py land create --name="MonSujet" --desc="..." --lang=fr
   docker compose exec mwi python mywi.py land addurl --land="MonSujet" --urls="https://example.org"
   docker compose exec mwi python mywi.py land crawl --name="MonSujet" --limit=10
   docker compose exec mwi python mywi.py land readable --name="MonSujet" --merge=smart_merge
   docker compose exec mwi python mywi.py land export --name="MonSujet" --type=pagecsv
   ```

Emplacements des données :
- Hôte : `${HOST_DATA_DIR}` (défaut `./data` dans le dépôt)
- Conteneur : `/app/data` (déjà utilisé par `settings.py`)

Extras optionnels :
- Installer les navigateurs Playwright (médias dynamiques) :
  ```bash
  docker compose exec mwi python install_playwright.py
  ```
- Construire avec les extras ML (FAISS, transformers) :
  ```bash
  MYWI_WITH_ML=1 docker compose build
  ```

Arrêter ou supprimer la stack :
```bash
docker compose down        # arrêter
```
```bash
docker compose down -v     # arrêter et supprimer les volumes (destructif)
```

### Docker CLI

1. Préparer un dossier hôte pour les données persistantes.
   ```bash
   mkdir -p ~/mywi_data
   ```
2. Builder l’image.
   ```bash
   docker build -t mwi:latest .
   ```
3. Lancer le conteneur en montant le dossier vers `/app/data` (chemin par défaut de `data_location`).
   ```bash
   docker run -dit --name mwi -v ~/mywi_data:/app/data mwi:latest
   ```
4. Ouvrir un shell et initialiser la base.
   ```bash
   docker exec -it mwi bash
   python mywi.py db setup
   ```
5. Utiliser ensuite le CLI (`python mywi.py ...`). Exporter `MYWI_DATA_DIR` si vous montez un autre chemin.

### Développement local

1. Cloner le dépôt et créer un environnement virtuel.
   ```bash
   git clone https://github.com/MyWebIntelligence/MyWebIntelligencePython.git
   cd MyWebIntelligencePython
   python -m venv venv
   source venv/bin/activate  # Windows : .\venv\Scripts\activate
   ```
2. Installer les dépendances.
   ```bash
   pip install -r requirements.txt
   ```
3. Configurer `data_location` dans `settings.py` vers un chemin absolu accessible (ex. `/Users/vous/mywi_data`).
4. Initialiser la base.
   ```bash
   python mywi.py db setup
   ```
5. (Optionnel) Installer les navigateurs Playwright pour l’analyse média dynamique.
   ```bash
   python install_playwright.py
   ```

## Guide CLI

MyWI expose des verbes sous la forme `python mywi.py <objet> <action>`. Tous les contrôleurs renvoient `1` en cas de succès (`0` sinon) pour faciliter l’automatisation.

### Lands & vocabulaire

- Créer un land :
  ```bash
  python mywi.py land create --name="MonSujet" --desc="..." --lang=fr
  ```
- Gérer le vocabulaire :
  ```bash
  python mywi.py land addterm --land="MonSujet" --terms="activisme, climat"
  python mywi.py land delterm --land="MonSujet" --terms="climat"
  ```
- Ajouter des URLs :
  ```bash
  python mywi.py land addurl --land="MonSujet" --urls="https://example.org,https://example.net"
  python mywi.py land addurl --land="MonSujet" --path=urls.txt
  ```
- Supprimer des données :
  ```bash
  python mywi.py land delete --name="MonSujet"
  python mywi.py land delete --name="MonSujet" --maxrel=0
  ```

### Collecte de données

- Crawler les URLs déjà stockées :
  ```bash
  python mywi.py land crawl --name="MonSujet" --limit=50 --depth=1
  python mywi.py land crawl --name="MonSujet" --http=503
  ```
  Options clés : `--limit`, `--depth`, `--http`, `--since`, `--async`.

- Amorcer depuis Google (SerpAPI) :
  ```bash
  python mywi.py land urlist --name="MonSujet" --query="(gilets jaunes) OR (manifestation)" \
    --datestart=2023-01-01 --dateend=2023-03-31 --timestep=week --lang=fr
  ```
  Nécessite `settings.serpapi_api_key` ou `MWI_SERPAPI_API_KEY`. Supporte `--sleep` pour la limitation de débit.

- Consolider / recalculer les heuristiques :
  ```bash
  python mywi.py land consolidate --name="MonSujet" --limit=100 --depth=2
  ```

### Contenu lisible

Utiliser le pipeline Mercury pour extraire un contenu propre, reconstruire les médias et recalculer la pertinence.

```bash
python mywi.py land readable --name="MonSujet" --limit=100 --merge=smart_merge
```

Options importantes :
- `--merge` : `smart_merge` (défaut), `mercury_priority`, `preserve_existing`
- `--depth` : restreindre à une profondeur
- `--force` : retraiter même si `readable_at` est déjà défini

Pré-requis : installer `@postlight/mercury-parser` (`npm install -g @postlight/mercury-parser`).

### Médias & domaines

- Analyser les médias :
  ```bash
  python mywi.py land medianalyse --name="MonSujet" --depth=2 --minrel=1
  python mywi.py land reanalyze --name="MonSujet" --limit=20
  python mywi.py land preview_deletion --name="MonSujet"
  python mywi.py land media_stats --name="MonSujet"
  ```
- Crawler les domaines indépendamment :
  ```bash
  python mywi.py domain crawl --limit=200 --http=404
  ```

L’analyse média tire parti de `python install_playwright.py` pour installer les navigateurs nécessaires à l’extraction dynamique.

### Exports & tags

- Générer des exports CSV, GEXF ou corpus :
  ```bash
  python mywi.py land export --name="MonSujet" --type=pagecsv
  python mywi.py land export --name="MonSujet" --type=nodegexf
  python mywi.py land export --name="MonSujet" --type=mediacsv
  python mywi.py land export --name="MonSujet" --type=corpus
  ```
  Types disponibles : `pagecsv`, `pagegexf`, `nodecsv`, `nodegexf`, `mediacsv`, `corpus`, `pseudolinks`.

- Travailler avec les tags qualitatifs :
  ```bash
  python mywi.py tag create --land="MonSujet" --name="thème/protestation"
  python mywi.py tag export --land="MonSujet" --type=matrix
  python mywi.py tag delete --land="MonSujet" --name="thème/protestation"
  ```

Les exports sont écrits sous `data/export_*` à l’intérieur du `data_location` configuré.

## Enrichissement & IA

### SEO Rank

Enrichir les expressions avec des métriques SEO Rank.

```bash
python mywi.py land seorank --name="MonSujet" --limit=100 --depth=1 --force
```

Configuration requise :
- Renseigner une clé API dans `settings.seorank_api_key` ou `MWI_SEORANK_API_KEY`.
- Options : `--http`, `--minrel`, `--request-delay`.

Par défaut, seules les expressions avec `http_status=200` et `relevance ≥ 1` sont ciblées. Utiliser `--force` pour rafraîchir les entrées possédant déjà un payload.

### Embeddings & similarité

Générer des embeddings par paragraphe et calculer les relations sémantiques.

```bash
python mywi.py embedding generate --name="MonSujet" --limit=500
python mywi.py embedding similarity --name="MonSujet" --method=cosine_lsh \
  --threshold=0.85 --lshbits=20 --topk=15 --minrel=1
python mywi.py embedding similarity --name="MonSujet" --method=nli --backend=bruteforce
```

Mise en place :
- Choisir un provider dans `settings.py` (`embed_provider`, `embed_model_name`, `embed_batch_size`).
- Installer les extras optionnels (`pip install -r requirements-ml.txt`, FAISS, transformers) pour NLI ou grandes exécutions.
- Exporter les pseudolinks :
  ```bash
  python mywi.py land export --name="MonSujet" --type=pseudolinks
  ```

### Validation LLM en masse

Demander à un LLM (OpenRouter) de valider la pertinence des pages par lot.

```bash
python mywi.py land llm validate --name="MonSujet" --limit=200 --force
```

Configuration :
- Activer la fonctionnalité dans `settings.py` (`openrouter_enabled=True`) ou via `MWI_OPENROUTER_ENABLED=true`.
- Fournir `openrouter_api_key`, `openrouter_model` et ajuster les paramètres (`openrouter_readable_min_chars`, `openrouter_readable_max_chars`, `openrouter_max_calls_per_run`).
- Exécuter `python mywi.py db migrate` si la base ne dispose pas encore des colonnes `validllm` / `validmodel`.

Le contrôleur enregistre `expression.validllm` (`"oui" / "non"`) et `expression.validmodel`. Un verdict `"non"` force `relevance=0`.

## Tests & qualité

- Lancer l’ensemble des tests :
  ```bash
  pytest tests/
  ```
- Focaliser sur le smoke test CLI :
  ```bash
  pytest tests/test_cli.py::test_functional_test
  ```
- Toute nouvelle logique mérite un test ciblé en suivant les patterns du dossier `tests/`.

En validation manuelle : créer un land, ajouter quelques URLs, exécuter `land crawl`, `land readable`, puis exporter `pagecsv`. Pour les changements liés aux médias, lancer `land medianalyse` sur une petite profondeur.

## Dépannage

- **Base absente ou corrompue** — lancer `python mywi.py db setup` (destructif) et vérifier `settings.data_location`.
- **Pas de contenu lisible** — vérifier l’installation de Mercury, tenter `land readable --merge=mercury_priority` ou `land consolidate`.
- **Exports vides** — contrôler `--minrel`, les filtres de profondeur et l’état `fetched_at/readable_at`.
- **Médias non détectés** — installer Playwright (`python install_playwright.py`) et vérifier les filtres d’extensions dans `settings.py`.
- **Timeouts HTTP** — ajuster `parallel_connections`, `default_timeout` ou relancer avec `land crawl --http=...`.
- **Erreurs API** — vérifier les clés SerpAPI / SEO Rank dans l’environnement ou `settings.py`.

## Licence

MyWI est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour les détails.
