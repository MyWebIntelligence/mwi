# My Web Intelligence (MyWI)

Version anglaise : [README.md](README.md)

MyWebIntelligence (MyWI) est un outil Python destiné aux équipes de recherche numérique. Il aide à constituer et analyser des corpus web, organisés par « lands » (projets thématiques). L’application collecte, nettoie et enrichit les pages, puis les stocke dans une base SQLite (mode WAL) facile à inspecter avec des outils comme [DB Browser for SQLite](https://sqlitebrowser.org/).

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
  - [Démarrage rapide : Docker Compose (recommandé)](#démarrage-rapide--docker-compose-recommandé)
  - [Docker manuel (avancé)](#docker-manuel-avancé)
  - [Installation locale](#installation-locale)
- [Scripts utiles](#scripts-utiles)
- [Utilisation](#utilisation)
  - [Notes générales](#notes-générales)
- [Gestion des lands](#gestion-des-lands)
  - [1. Créer un land](#1-créer-un-land)
  - [2. Lister les lands](#2-lister-les-lands)
  - [3. Ajouter des termes](#3-ajouter-des-termes)
  - [4. Ajouter des URLs](#4-ajouter-des-urls)
  - [5. Récupérer des URLs via SerpAPI](#5-récupérer-des-urls-via-serpapi)
  - [6. Supprimer un land ou des expressions](#6-supprimer-un-land-ou-des-expressions)
- [Collecte de données](#collecte-de-données)
  - [1. Crawler les URLs du land](#1-crawler-les-urls-du-land)
  - [2. Extraire un contenu lisible (pipeline Mercury)](#2-extraire-un-contenu-lisible-pipeline-mercury)
  - [3. Capturer les métriques SEO Rank](#3-capturer-les-métriques-seo-rank)
  - [4. Analyse médias](#4-analyse-médias)
  - [5. Crawl des domaines](#5-crawl-des-domaines)
- [Exports](#exports)
  - [1. Exporter un land](#1-exporter-un-land)
  - [2. Exporter les tags](#2-exporter-les-tags)
- [Mettre à jour les domaines depuis les heuristiques](#mettre-à-jour-les-domaines-depuis-les-heuristiques)
- [Pipeline de consolidation des lands](#pipeline-de-consolidation-des-lands)
- [Tests](#tests)
- [Embeddings & pseudolinks (guide utilisateur)](#embeddings--pseudolinks-guide-utilisateur)
  - [Objectif](#objectif)
  - [Pré-requis & installation](#pré-requis--installation)
  - [Modèles](#modèles)
  - [Paramètres (référence)](#paramètres-référence)
  - [Commandes & paramètres](#commandes--paramètres)
  - [Dépannage & précautions](#dépannage--précautions)
  - [Bonnes pratiques — performance](#bonnes-pratiques--performance)
  - [Choix des modèles et recours](#choix-des-modèles-et-recours)
  - [Progression & logs](#progression--logs)
  - [Méthodes de similarité](#méthodes-de-similarité)
  - [Choisir le backend ANN (FAISS)](#choisir-le-backend-ann-faiss)
  - [Similarité scalable (lands volumineux)](#similarité-scalable-lands-volumineux)
  - [Relations NLI (ANN + cross-encoder)](#relations-nli-ann--cross-encoder)
- [Dépannage & réparation](#dépannage--réparation)
  - [Garder le schéma de base à jour](#garder-le-schéma-de-base-à-jour)
  - [Récupération SQLite](#récupération-sqlite)
- [Pour les développeurs](#pour-les-développeurs)
  - [Architecture & flux internes](#architecture--flux-internes)
  - [Schéma de données (SQLite/Peewee)](#schéma-de-données-sqlitepeewee)
  - [Workflows principaux](#workflows-principaux)
  - [Notes d’implémentation](#notes-dimplémentation)
  - [Paramètres](#paramètres)
  - [Tests](#tests-1)
  - [Extension](#extension)
- [Licence](#licence)

## Fonctionnalités

- **Lands thématiques** : organisez URLs, lexiques et exports par projet.
- **Crawl résilient** : parallélisme contrôlé, retries, filtres HTTP, profondeur maîtrisée.
- **Extraction Mercury** : contenu lisible propre avec fusion configurable, enrichissement des métadonnées, recalcul de la pertinence.
- **Analyse médias** : dimensions, formats, couleurs dominantes, EXIF, hash perceptuel, score NSFW, erreurs traçables.
- **Enrichissements** : SerpAPI pour préremplir les lands, SEO Rank pour les métriques, validation LLM (OpenRouter) en option.
- **Embeddings & pseudolinks** : vecteurs par paragraphe, similarité cosine (exacte ou LSH), pipeline NLI pour qualifier les relations logiques.
- **Exports multiples** : CSV, GEXF (pages/nœuds), corpus brut, médias, tags, pseudolinks.
- **Configuration centralisée** : `settings.py` + variables d’environnement pour adapter timeouts, clés API, heuristiques, providers ML.

---

# Installation

## Démarrage rapide : Docker Compose (recommandé)

**Commande unique**

```bash
./scripts/docker-compose-setup.sh [basic|api|llm]
```
Si vous omettez l’argument, le script utilise `basic`. Choisissez `api` pour configurer SerpAPI / SEO Rank / OpenRouter, ou `llm` pour inclure en plus les dépendances embeddings & NLI.

Sous Windows, exécutez ce script dans un terminal compatible Bash :
- Git Bash : `./scripts/docker-compose-setup.sh`
- PowerShell : `& "C:\Program Files\Git\bin\bash.exe" ./scripts/docker-compose-setup.sh`
- WSL : `wsl bash ./scripts/docker-compose-setup.sh`
Un double-clic sur le fichier `.sh` ne lance rien.

**Approche pas-à-pas**

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/MyWebIntelligence/mwi.git
   cd mwi
   ```
2. Générer `.env` avec l’assistant interactif :
   ```bash
   python scripts/install-docker-compose.py
   ```
   (Sous Windows, `py -3 scripts/install-docker-compose.py` fonctionne aussi.)
3. Construire et démarrer le conteneur :
   ```bash
   docker compose up -d --build
   ```
4. Créer `settings.py` **depuis le conteneur** (à faire une seule fois par environnement) :
   ```bash
   docker compose exec mwi bash -lc "cp settings-example.py settings.py"
   ```
   Pour personnaliser la configuration, lancez plutôt :
   ```bash
   docker compose exec -it mwi python scripts/install-basic.py --output settings.py
   ```
5. Initialiser puis vérifier la base :
   ```bash
   docker compose exec mwi python mywi.py db setup
   docker compose exec mwi python mywi.py land list
   ```

> ⚠️ `settings.py` n’est jamais créé automatiquement dans le conteneur. Copiez `settings-example.py` (ou exécutez `python scripts/install-basic.py`) avant de lancer les commandes MyWI pour y renseigner chemins, clés API et options spécifiques.

**Où sont vos données ?**

- Machine hôte : `./data` (ou `HOST_DATA_DIR` défini dans `.env`).
- Conteneur : `/app/data` (via `MYWI_DATA_DIR`).

**Commandes de gestion**

```bash
docker compose up -d        # Démarrer
docker compose down         # Arrêter
docker compose logs mwi     # Voir les logs
docker compose exec mwi bash  # Entrer dans le conteneur
```

## Docker manuel (avancé)

```bash
# Construction
docker build -t mwi:latest .

# Exécution
docker run -dit --name mwi -v ~/mywi_data:/app/data mwi:latest

# Création de settings.py dans le conteneur (premier lancement)
docker exec mwi bash -lc "cp settings-example.py settings.py"
# Variante interactive :
# docker exec -it mwi python scripts/install-basic.py --output settings.py

# Initialisation
docker exec -it mwi python mywi.py db setup

# Utilisation
docker exec -it mwi python mywi.py land list
```

Gestion : `docker stop mwi` · `docker start mwi` · `docker rm mwi`.

## Installation locale

**Pré-requis** : Python 3.10+, pip, git.

```bash
# 1. Cloner et créer un environnement virtuel
git clone https://github.com/MyWebIntelligence/mwi.git
cd mwi
python3 -m venv .venv
# Windows PowerShell : py -3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell : .\.venv\Scripts\Activate.ps1 ; cmd.exe : .\.venv\Scripts\activate.bat

# 2. Configurer (assistant)
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python scripts/install-basic.py

# 3. Initialiser la base
python mywi.py db setup

# 4. Vérifier
python mywi.py land list
```

**Étapes optionnelles**

- APIs : `python scripts/install-api.py`
- Embeddings/LLM : `python -m pip install -r requirements-ml.txt && python scripts/install-llm.py`
- Médias dynamiques (Playwright) :
  - Navigateurs : `python install_playwright.py`
  - Dépendances Debian/Ubuntu : `sudo apt-get install libnspr4 libnss3 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2`
  - Docker : `docker compose exec mwi bash -lc "apt-get update && apt-get install -y <libs>"` puis `docker compose exec mwi python install_playwright.py`

**Problèmes NLTK (Windows/macOS)**

```bash
python -m nltk.downloader punkt punkt_tab
# En cas d’erreur SSL : pip install certifi
```

## Scripts utiles

**Démarrage express**
- `scripts/docker-compose-setup.sh` — bootstrap complet Docker (crée/backup `.env`, lance l’assistant, build, démarre, initialise la base, tests optionnels). `./scripts/docker-compose-setup.sh [basic|api|llm]`.

**Assistants interactifs**
- `scripts/install-docker-compose.py` — génère `.env` pour Compose (fuseau horaire, mapping dossier hôte ↔ `/app/data`, flags Playwright/ML, clés SerpAPI/SEO Rank/OpenRouter, paramètres embeddings/NLI). `python scripts/install-docker-compose.py [--level basic|api|llm] [--output .env]`.
- `scripts/install-basic.py` — produit un `settings.py` minimal (stockage, timeouts, parallélisme, user agent, médias dynamiques, analyse médias, heuristiques). `python scripts/install-basic.py [--output settings.py]`.
- `scripts/install-api.py` — enregistre les clés SerpAPI / SEO Rank / OpenRouter dans `settings.py` (avec fallback via variables d’environnement). `python scripts/install-api.py [--output settings.py]`.
- `scripts/install-llm.py` — configure provider d’embeddings, modèles/backends NLI, paramètres de retry/batching (vérifie les dépendances ML). `python scripts/install-llm.py [--output settings.py]`.

**Diagnostics & reprise**
- `scripts/test-apis.py` — teste SerpAPI, SEO Rank, OpenRouter (`--serpapi`, `--seorank`, `--openrouter`, `--all`, `-v` pour le détail). `python scripts/test-apis.py ...`.
- `scripts/sqlite_recover.sh` — réparation SQLite non destructive (voir [Récupération SQLite](#récupération-sqlite)). `scripts/sqlite_recover.sh [INPUT_DB] [OUTPUT_DB]`.

**Utilitaires**
- `scripts/install-nltk.py` — télécharge `punkt` et `punkt_tab` pour NLTK. `python scripts/install-nltk.py`.
- `scripts/crawl_robuste.sh` — exemple de boucle `land crawl` avec retries (éditer nom du land et paramètres). `bash scripts/crawl_robuste.sh`.
- `scripts/install_utils.py` — helpers communs aux assistants (non exécutable seul).

# Utilisation

## Notes générales

- Toutes les commandes passent par `python mywi.py ...`.
- En Docker :

```bash
# Vérifier que le service tourne
docker compose up -d
# Shell dans le conteneur
docker compose exec mwi bash
# ou
docker exec -it mwi bash
# >>> prompt ≈ root@<container_id>:/app#

# Exemple de commande applicative
python mywi.py land list
```

- En local : activez votre environnement virtuel avant d’appeler la CLI.

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat

python mywi.py land list
```

- Remplacez les placeholders (`LAND`, `TERMS`, `https://…`) par vos valeurs.

---

## Gestion des lands

### 1. Créer un land

```bash
python mywi.py land create --name="MonProjet" --desc="Description" --lang=fr
```

### 2. Lister les lands

```bash
python mywi.py land list
python mywi.py land list --name="MonProjet"
```

### 3. Ajouter des termes

```bash
python mywi.py land addterm --land="MonProjet" --terms="mot1, mot2"
```

### 4. Ajouter des URLs

```bash
# Directement
python mywi.py land addurl --land="MonProjet" --urls="https://exemple.org,https://exemple.net"

# Depuis un fichier
python mywi.py land addurl --land="MonProjet" --path=urls.txt
```

### 5. Récupérer des URLs via SerpAPI

```bash
python mywi.py land urlist --name="MonProjet" --query="(mot clé)" \
  --datestart=2023-01-01 --dateend=2023-03-31 --timestep=week --lang=fr
```

- Nécessite `settings.serpapi_api_key` ou `MWI_SERPAPI_API_KEY`.
- `--sleep` contrôle la pause (défaut : 1 s).

### 6. Supprimer un land ou des expressions

```bash
python mywi.py land delete --name="MonProjet"
python mywi.py land delete --name="MonProjet" --maxrel=0.5
```

---

## Collecte de données

### 1. Crawler les URLs du land

```bash
python mywi.py land crawl --name="MonProjet" [--limit N] [--http CODE] [--depth D]
```

- `--limit` : plafond d’URLs par run.
- `--http` : relancer uniquement les codes spécifiés (`--http 503`).
- `--depth` : limite la profondeur.

> Astuce shell :
> `for i in {1..100}; do python mywi.py land crawl --name="MonProjet" --depth=0 --limit=100; done`

### 2. Extraire un contenu lisible (pipeline Mercury)

**Pré-requis** : `npm install -g @postlight/mercury-parser`

```bash
python mywi.py land readable --name="MonProjet" [--limit N] [--depth D] [--merge stratégie] [--llm=true|false]
```

- `smart_merge` (défaut) : fusion intelligente.
- `mercury_priority` : Mercury écrase tout.
- `preserve_existing` : complète uniquement les champs vides.
- `--llm=true` : filtre OpenRouter (si configuré).

### 3. Capturer les métriques SEO Rank

```bash
python mywi.py land seorank --name="MonProjet" [--limit N] [--depth D] [--force]
```

- Clé API : `settings.seorank_api_key` ou `MWI_SEORANK_API_KEY`.
- Par défaut : HTTP 200 et `relevance ≥ 1`.
- `--force` : rafraîchit même les entrées existantes.

### 4. Analyse médias

```bash
python mywi.py land medianalyse --name="MonProjet" [--depth D] [--minrel R]
```

Télécharge, mesure (dimensions/taille), extrait couleurs & EXIF, calcule hash, NSFW, consigne les erreurs.

### 5. Crawl des domaines

```bash
python mywi.py domain crawl [--limit N] [--http CODE]
```

---

## Exports

### 1. Exporter un land

```bash
python mywi.py land export --name="MonProjet" --type=pagecsv
python mywi.py land export --name="MonProjet" --type=nodegexf
python mywi.py land export --name="MonProjet" --type=mediacsv
python mywi.py land export --name="MonProjet" --type=corpus
python mywi.py land export --name="MonProjet" --type=pseudolinks
```

Types : `pagecsv`, `fullpagecsv`, `nodecsv`, `pagegexf`, `nodegexf`, `mediacsv`, `corpus`, `pseudolinks`, `pseudolinkspage`, `pseudolinksdomain`.

### 2. Exporter les tags

```bash
python mywi.py tag export --name="MonProjet" --type=matrix
python mywi.py tag export --name="MonProjet" --type=content
```

---

## Mettre à jour les domaines depuis les heuristiques

```bash
python mywi.py heuristic update
```

## Pipeline de consolidation des lands

```bash
python mywi.py land consolidate --name="MonProjet" [--limit N] [--depth D]
```

## Tests

```bash
pytest tests/
pytest tests/test_cli.py
pytest tests/test_cli.py::test_functional_test
```

---

# Embeddings & pseudolinks (guide utilisateur)

## Objectif

- Générer des embeddings par paragraphe.
- Relier les paragraphes proches (pseudolinks) et, si besoin, qualifier la relation via NLI (entailment / neutral / contradiction).
- Exporter les relations au niveau paragraphe, page ou domaine.

## Pré-requis & installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

Option ML :

```bash
python -m pip install -r requirements-ml.txt
```

Vérification : `python mywi.py embedding check`

## Modèles

- NLI recommandé : `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
- Fallback léger (EN) : `typeform/distilbert-base-uncased-mnli`.

## Paramètres (référence)

- Embeddings : `embed_provider`, `embed_model_name`, `embed_batch_size`, `embed_min_paragraph_chars`, `embed_max_paragraph_chars`, `embed_similarity_method`, `embed_similarity_threshold`.
- Rappel ANN / NLI : `similarity_backend`, `similarity_top_k`, `nli_model_name`, `nli_fallback_model_name`, `nli_backend_preference`, `nli_batch_size`, `nli_max_tokens`, `nli_torch_num_threads`, `nli_progress_every_pairs`, `nli_show_throughput`.
- Variables CPU : `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TOKENIZERS_PARALLELISM=false`.

## Commandes & paramètres

```bash
python mywi.py embedding generate --name=LAND [--limit N]

python mywi.py embedding similarity --name=LAND --method=cosine \
  --threshold=0.85 [--minrel R]

python mywi.py embedding similarity --name=LAND --method=cosine_lsh \
  --lshbits=20 --topk=15 --threshold=0.85 [--minrel R] [--maxpairs M]

python mywi.py embedding similarity --name=LAND --method=nli \
  --backend=faiss|bruteforce --topk=10 [--minrel R] [--maxpairs M]

python mywi.py land export --name=LAND --type=pseudolinks
python mywi.py land export --name=LAND --type=pseudolinkspage
python mywi.py land export --name=LAND --type=pseudolinksdomain
```

## Dépannage & précautions

- `score_raw=0.5` + `score=0` : fallback neutre → installer les dépendances ML ou choisir un autre modèle.
- Colonne `score_raw` absente : `python mywi.py db migrate`.
- Segfault macOS (OpenMP/Torch) : venv pip-only, commencer avec `OMP_NUM_THREADS=1`, augmenter ensuite.
- Lenteur : diminuer `nli_batch_size`, filtrer `--minrel`, plafonner `--maxpairs`, ajuster les threads.
- Trop de paires : augmenter `--threshold`, `--lshbits`, réduire `--topk`, utiliser `--minrel`.

## Bonnes pratiques — performance

- ≤ 50k paragraphes : `--method=cosine --threshold=0.85 --minrel=1`.
- ≥ 100k paragraphes : `--method=cosine_lsh`, `--lshbits=18–22`, `--topk=10–20`, `--threshold≥0.85`, `--maxpairs` pour plafonner.
- Pipeline NLI : FAISS recommandé, départ `--topk=6–10`, `--minrel=1–2`, `--maxpairs=20k–200k`, ajuster `nli_batch_size` (32–96) et `nli_max_tokens` (384–512).

## Choix des modèles et recours

- Par défaut : DeBERTa multilingue (`sentencepiece` requis).
- Alternative sûre (EN) : DistilBERT MNLI.
- Sans dépendances ML : fallback neutre (`score=0`).

## Progression & logs

- Rappel ANN : journalise les candidats.
- NLI : affiche `pairs/s`, ETA, cumul.
- Résumé final : temps total et volume traité.

## Méthodes de similarité

- `cosine` : comparaison exacte O(n²).
- `cosine_lsh` : approximation scalable via LSH.
- `nli` : ANN + cross-encoder, scores ∈ {-1,0,1}.

## Choisir le backend ANN (FAISS)

- Installer FAISS : `pip install faiss-cpu`.
- Forcer `--backend=faiss` ou `--backend=bruteforce`.
- Paramètre global : `similarity_backend = 'faiss'` dans `settings.py`.
- Sans FAISS : fallback bruteforce.

## Similarité scalable (lands volumineux)

```bash
python mywi.py embedding similarity \
  --name=LAND \
  --method=cosine_lsh \
  --threshold=0.85 \
  --lshbits=20 \
  --topk=15 \
  --minrel=1 \
  --maxpairs=5000000
```

- `--lshbits` : plus élevé → buckets plus fins.
- `--topk` : voisins conservés.
- `--threshold` : seuil minimal.
- `--minrel` : filtre pertinence.
- `--maxpairs` : limite globale.

## Relations NLI (ANN + cross-encoder)

```bash
pip install sentence-transformers transformers
pip install faiss-cpu

python mywi.py embedding similarity \
  --name=LAND \
  --method=nli \
  --backend=bruteforce \
  --topk=50 \
  --minrel=1 \
  --maxpairs=2000000
```

Paramètres clés : `nli_model_name`, `nli_batch_size`, `similarity_backend`, `similarity_top_k`.

---

# Dépannage & réparation

## Garder le schéma de base à jour

```bash
python mywi.py db migrate
cp data/mwi.db data/mwi.db.bak_$(date +%Y%m%d_%H%M%S)
```

## Récupération SQLite

```bash
chmod +x scripts/sqlite_recover.sh
scripts/sqlite_recover.sh data/mwi.db data/mwi_repaired.db
```

- Sauvegarde la base et les fichiers `-wal`/`-shm`.
- Tente `.recover`, fallback `.dump`.
- Reconstruit `data/mwi_repaired.db`, éxécute `PRAGMA integrity_check;` et liste les tables.
- Tester avant remplacement :

```bash
mkdir -p data/test-repaired
cp data/mwi_repaired.db data/test-repaired/mwi.db
MYWI_DATA_DIR="$PWD/data/test-repaired" python mywi.py land list
```

---

# Pour les développeurs

## Architecture & flux internes

```
mywi.py  →  mwi/cli.py  →  mwi/controller.py  →  mwi/core.py & mwi/export.py
                                     ↘ mwi/model.py (Peewee)
                                     ↘ mwi/readable_pipeline.py
                                     ↘ mwi/media_analyzer.py
                                     ↘ mwi/embedding_pipeline.py
```

- `mywi.py` : point d’entrée CLI.
- `mwi/cli.py` : parsing (`argparse`), expose `command_run()`.
- `mwi/controller.py` : façade, renvoie 1 (succès) ou 0 (échec).
- `mwi/core.py` : crawl, pipeline Mercury, heuristiques, consolidation, médias.
- `mwi/export.py` : exports CSV/GEXF/corpus.
- `mwi/model.py` : schéma Peewee, pragmas SQLite.

## Schéma de données (SQLite/Peewee)

- `Land`, `Word`, `LandDictionary`, `Domain`, `Expression`, `ExpressionLink`, `Media`, `Paragraph`, `ParagraphEmbedding`, `ParagraphSimilarity`, `Tag`, `TaggedContent`.

## Workflows principaux

- Initialisation : `python mywi.py db setup`
- Cycle land : créer → ajouter termes/URLs → `land crawl` → `land readable` → exports
- Médias : `python mywi.py land medianalyse ...`
- SEO Rank : `python mywi.py land seorank ...`
- Domaines : `python mywi.py domain crawl`
- Tags : `python mywi.py tag export`
- Embeddings : `python mywi.py embedding generate`, `python mywi.py embedding similarity`

## Notes d’implémentation

- Pertinence basée sur les lemmes (titre + contenu).
- Crawl asynchrone avec contrôle du parallélisme, timeouts, retries, archivage HTML optionnel.
- Médias : association automatique, filtrage configurable, hash, erreurs persistées.
- Exports : requêtes Peewee/SQL ciblées, génération CSV/GEXF enrichie.

## Paramètres

- `data_location`, `archive`, `dynamic_media_extraction`, `parallel_connections`, `default_timeout`, `user_agent`, `heuristics`.
- Embeddings : `embed_provider`, `embed_model_name`, `embed_api_url`, `embed_batch_size`, `embed_min_paragraph_chars`, `embed_max_paragraph_chars`, `embed_similarity_method`, `embed_similarity_threshold`, retrys.
- OpenRouter : `openrouter_enabled`, `openrouter_api_key`, `openrouter_model`, `openrouter_timeout`, `openrouter_readable_min_chars`, `openrouter_readable_max_chars`, `openrouter_max_calls_per_run`.
- SEO Rank : `seorank_api_base_url`, `seorank_api_key`, `seorank_timeout`, `seorank_request_delay`.
- SerpAPI : `serpapi_api_key`, `serpapi_base_url`, `serpapi_timeout`.
- NLI : `nli_model_name`, `nli_fallback_model_name`, `nli_backend_preference`, `nli_batch_size`, `nli_max_tokens`, `nli_torch_num_threads`, `nli_progress_every_pairs`, `nli_show_throughput`, `nli_entailment_threshold`, `nli_contradiction_threshold`.
- Similarité : `similarity_backend`, `similarity_top_k`.

## Tests

- `pytest tests/`
- `pytest tests/test_cli.py`

## Extension

- Nouvel export : étendre `mwi/export.py`, raccorder dans `controller.py`.
- Provider embeddings : implémenter dans `embedding_pipeline.py`, déclarer dans `settings.py`.
- Enrichissement API : ajouter un contrôleur, la configuration `settings.py` et les scripts d’installation.

---

# Licence

Projet distribué sous licence MIT — voir [LICENSE](LICENSE).
