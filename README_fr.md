# My Web Intelligence (MyWI)

Version anglaise : [README.md](README.md)

MyWebIntelligence (MyWI) est un outil Python destiné aux équipes en humanités numériques qui doivent constituer et analyser rapidement des corpus web. Le projet regroupe les URLs, contenus, médias, métriques et annotations dans une base SQLite, organisée par « lands » (projets thématiques). La CLI orchestre les collectes, traitements et exports pour transformer des listes d’URLs en jeux de données exploitables.

Toutes les données sont stockées dans SQLite (mode WAL), ce qui facilite l’inspection avec des outils comme [DB Browser for SQLite](https://sqlitebrowser.org/) ou l’intégration dans des tableaux de bord.

## Sommaire
- [Fonctionnalités](#fonctionnalités)
- [Architecture & internes](#architecture--internes)
- [Utiliser Docker Compose (recommandé)](#utiliser-docker-compose-recommandé)
- [Utiliser Docker (manuel)](#utiliser-docker-manuel)
- [Installation locale](#installation-locale)
- [Notes générales](#notes-générales)
- [Gestion des lands](#gestion-des-lands)
- [Collecte de données](#collecte-de-données)
- [Export de données](#export-de-données)
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
  - [Récupération SQLite](#récupération-sqlite)
- [Licence](#licence)

## Fonctionnalités

- Gestion des lands : créez des espaces de travail thématiques avec vocabulaire, URLs, exports et métriques isolés.
- Crawl résilient : parallélisme poli, filtrage par statut HTTP, profondeur contrôlée, retries et archivage HTML optionnel.
- Extraction lisible de haute qualité via le pipeline Mercury (stratégies de fusion, enrichissement des métadonnées, reconstruction des liens et recalcul de la pertinence).
- Analyse médias pour images, vidéos et audio (métadonnées, hachages, détection de doublons, couleurs dominantes, score NSFW, erreurs d’analyse traçables).
- Détection renforcée des fichiers médias (extensions en majuscules/minuscules, contenus dynamiques via Playwright en option).
- Analyse des domaines avec crawl dédié, heuristiques et contrôleurs spécialisés.
- Exports riches : CSV, GEXF (pages ou nœuds), listing médias, matrices de tags, corpus brut zippé, pseudolinks.
- Embeddings par paragraphe avec similarité cosine (exacte ou LSH) et pipeline NLI pour qualifier les relations logiques.
- Enrichissement SEO : génération de listes d’URLs via SerpAPI, métriques SEO Rank stockées en JSON par expression.
- Validation LLM : filtre OpenRouter en ligne (oui/non) et validation de masse, désactivable pour retrouver le comportement historique.
- Paramétrage centralisé dans `settings.py`, surchargé si besoin par des variables d’environnement (timeouts, providers, clés API, heuristiques).

---

## Architecture & internes

### Structure des fichiers et flux

```
mywi.py  →  mwi/cli.py  →  mwi/controller.py  →  mwi/core.py & mwi/export.py
                                     ↘ mwi/model.py (Peewee ORM)
                                     ↘ mwi/readable_pipeline.py
                                     ↘ mwi/media_analyzer.py
                                     ↘ mwi/embedding_pipeline.py
```

- `mywi.py` : point d’entrée CLI.
- `mwi/cli.py` : construit la CLI (`argparse`), expose `command_run()` pour les tests.
- `mwi/controller.py` : façade qui route chaque verbe vers `core.py`, `export.py` ou `model.py` et renvoie `1` (succès) ou `0` (échec).
- `mwi/core.py` : cœur métier (crawl, parsing, pipeline Mercury, heuristiques, consolidation, gestion médias).
- `mwi/export.py` : centralise les exports CSV/GEXF/corpus.
- `mwi/model.py` : schéma Peewee (SQLite), indexes, relations.

### Modèle de données (SQLite via Peewee)

- **Land** : métadonnées du projet (nom, langue, description, timestamps).
- **Word** / **LandDictionary** : vocabulaire normalisé et liaison many-to-many avec les lands.
- **Domain** : informations agrégées par domaine (titre, meta description, statut HTTP, dernier crawl).
- **Expression** : page individuelle (URL, profondeur, contenu lisible, score, SEO Rank JSON, verdict LLM, timestamps).
- **ExpressionLink** : liens hypertextes orientés entre expressions.
- **Media** : médias découverts dans les expressions (type, dimensions, format, hash perceptuel, NSFW, erreurs).
- **Paragraph**, **ParagraphEmbedding**, **ParagraphSimilarity** : texte segmenté par paragraphe, vecteurs, liens sémantiques.
- **Tag** / **TaggedContent** : taxonomie qualitative et extraits associés.

### Workflows principaux

- **Initialisation** : `python mywi.py db setup`
- **Cycle d’un land** : créer le land → ajouter des termes/URLs → `land crawl` → `land readable` → exports → nettoyage éventuel.
- **Analyse médias** : `python mywi.py land medianalyse --name=LAND [--depth D] [--minrel R]`
- **Enrichissement SEO Rank** : `python mywi.py land seorank --name=LAND [--limit N] [--depth D] [--force]`
- **Gestion des domaines** : `python mywi.py domain crawl`
- **Tags** : `python mywi.py tag export`
- **Mises à jour heuristiques** : `python mywi.py heuristic update`
- **Embeddings & similarité** :
  - Génération : `python mywi.py embedding generate --name=LAND [--limit N]`
  - Similarité : `python mywi.py embedding similarity --name=LAND --method=cosine|cosine_lsh|nli`

### Notes d’implémentation

- **Pertinence** : score basé sur les occurrences pondérées de lemmes dans le titre et le contenu.
- **Crawl** : exécutions asynchrones polies (contrôle du parallélisme, timeouts, retries).
- **Extraction médias** : liaisons automatiques, stockage sélectif (configurable), détection des doublons.
- **Exports** : requêtes Peewee/SQL dynamiques, génération CSV et GEXF avec attributs enrichis.

### Paramètres clés (`settings.py`)

- `data_location`, `user_agent`, `parallel_connections`, `default_timeout`, `archive`, `heuristics`.

#### Embeddings — configuration
- `embed_provider` : `fake`, `http`, `openai`, `mistral`, `gemini`, `huggingface`, `ollama`.
- `embed_api_url` : endpoint pour les providers HTTP génériques (`POST {"model": name, "input": [texts...]}`).
- `embed_model_name` : libellé conservé avec les vecteurs.
- `embed_batch_size` : taille de lot envoyée au provider.
- `embed_min_paragraph_chars` / `embed_max_paragraph_chars` : longueur minimale/maximale d’un paragraphe.
- `embed_similarity_method` / `embed_similarity_threshold` : méthode et seuil pour la similarité.

Providers spécifiques :
- **OpenAI / Mistral** : payload `{ "model": name, "input": [texts...] }` → réponse `{ "data": [{"embedding": [...]}, ...] }`.
- **Gemini** : endpoint `:batchEmbedContents` → `{ "embeddings": [{"values": [...]}, ...] }`.
- **Hugging Face** : `{ "inputs": [texts...] }` → liste de vecteurs.
- **Ollama** : appels séquentiels sur `/api/embeddings` (`{"model": name, "prompt": text}`), pas de batch.

#### Option : Filtre LLM (OpenRouter oui/non)

- Variables : `MWI_OPENROUTER_ENABLED`, `MWI_OPENROUTER_API_KEY`, `MWI_OPENROUTER_MODEL`, `MWI_OPENROUTER_TIMEOUT`, `MWI_OPENROUTER_READABLE_MAX_CHARS`, `MWI_OPENROUTER_MAX_CALLS_PER_RUN`.
- Lorsqu’il est désactivé ou non configuré, le pipeline revient au comportement traditionnel.

#### Validation LLM en masse (oui/non)

Commande :
```bash
python mywi.py land llm validate --name=LAND [--limit N] [--force]
```

Pré-requis :
- `settings.py` : `openrouter_enabled=True`, clé API et modèle renseignés.
- Base ancienne ? Lancer `python mywi.py db migrate` pour ajouter les colonnes `validllm` / `validmodel`.

Comportement :
- Récupère les expressions sans verdict et demande au LLM "oui/non".
- Renseigne `expression.validllm` (`"oui"|"non"`) et `expression.validmodel` (slug du modèle).
- Filtre sur les expressions avec contenu lisible non nul et longueur ≥ `openrouter_readable_min_chars`.
- Respecte `openrouter_readable_max_chars` et `openrouter_max_calls_per_run`.
- Si verdict `"non"`, force `relevance = 0`.
- `--force` : inclut aussi les entrées marquées `"non"` pour revalidation.

#### Enrichissement SEO Rank

- `python mywi.py land seorank --name=LAND [--limit N] [--depth D] [--force]`
- Clé à fournir dans `settings.seorank_api_key` (ou `MWI_SEORANK_API_KEY`).
- Paramètres : `seorank_api_base_url`, `seorank_timeout`, `seorank_request_delay`.
- Par défaut, seules les expressions `http_status = 200` et `relevance ≥ 1` sont traitées. Utilisez `--http=all` ou `--minrel=0` pour élargir.

#### Bootstrap SerpAPI (`land urlist`)

- `python mywi.py land urlist --name=LAND --query="..." [--datestart AAAA-MM-JJ --dateend AAAA-MM-JJ --timestep week]`
- Config : `serpapi_api_key`, `serpapi_base_url`, `serpapi_timeout`, `--sleep` (secondes entre pages).

---

## Utiliser Docker Compose (recommandé)

Docker Compose est la manière la plus simple d’exécuter MyWI : tout reste isolé dans un conteneur et vos données demeurent sur votre machine. Les étapes ci-dessous supposent que vous débutez avec Docker.

### Prérequis
- Installer [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Compose est inclus) et le lancer avant de poursuivre.

### Étape 1 – Préparer les fichiers de configuration
1. Copier `.env.example` vers `.env`.
   ```bash
   cp .env.example .env
   ```
   - Conserver `HOST_DATA_DIR=./data` pour stocker la base et les exports dans le dépôt, ou remplacer par un chemin absolu (ex. `HOST_DATA_DIR=/Users/vous/mywi_data` sous macOS/Linux ou `HOST_DATA_DIR=C:/Users/vous/mywi_data` sous Windows).
2. Copier le fichier d’exemple des réglages.
   ```bash
   cp settings-example.py settings.py
   ```
   - Ajouter vos clés API (SerpAPI, SEO Rank, OpenRouter, embeddings…) dans `settings.py`, ou bien les renseigner dans `.env` via des variables comme `MWI_SERPAPI_API_KEY=...` qui seront transmises automatiquement au conteneur.

### Étape 2 – Construire et démarrer les services
```bash
docker compose up -d --build
```
Le premier lancement télécharge les dépendances et peut durer plusieurs minutes. Pour les lancements suivants, `docker compose up -d` suffit.

### Étape 3 – Initialiser la base de données (une seule fois)
```bash
docker compose exec mwi python mywi.py db setup
```

### Étape 4 – Lancer les commandes CLI
Utiliser `docker compose exec mwi python mywi.py ...` pour chaque commande. Exemples :
```bash
docker compose exec mwi python mywi.py land create --name="MonSujet" --desc="…" --lang=fr
docker compose exec mwi python mywi.py land addurl --land="MonSujet" --urls="https://example.org"
docker compose exec mwi python mywi.py land crawl --name="MonSujet" --limit=10
docker compose exec mwi python mywi.py land readable --name="MonSujet" --merge=smart_merge
docker compose exec mwi python mywi.py land export --name="MonSujet" --type=pagecsv
```

### Étape 5 – Extras optionnels
- Installer Playwright pour l’extraction dynamique :
  ```bash
  docker compose exec mwi python install_playwright.py
  ```
- Inclure les dépendances ML (FAISS + transformers) si vous prévoyez d’utiliser les embeddings/NLI en local :
  ```bash
  MYWI_WITH_ML=1 docker compose build
  docker compose up -d
  ```

### Étape 6 – Arrêter ou réinitialiser l’environnement
```bash
docker compose down          # arrêter le conteneur
docker compose down -v       # arrêter et supprimer le volume Docker (DETRUIT la base)
```

#### Où sont mes données ?
- Sur la machine hôte : le dossier indiqué par `HOST_DATA_DIR` (défaut : `./data` dans le dépôt).
- Dans le conteneur : `/app/data`. `settings.py` pointe déjà vers cet emplacement, aucune configuration supplémentaire n’est nécessaire.

## Utiliser Docker (manuel)

Choisissez cette option si vous préférez saisir vous-même les commandes Docker plutôt que d’utiliser Docker Compose.

### Prérequis
- Installer [Docker Desktop](https://www.docker.com/products/docker-desktop/) ou Docker Engine et vérifier qu’il s’exécute.

### Étape 1 – Cloner le dépôt
```bash
git clone https://github.com/MyWebIntelligence/MyWebIntelligencePython.git
cd MyWebIntelligencePython
```

### Étape 2 – Préparer le fichier `settings.py`
```bash
cp settings-example.py settings.py
```
- Stocker vos clés API (SerpAPI, SEO Rank, OpenRouter, embeddings…) dans `settings.py` ou prévoir de les passer au conteneur via `-e MWI_SERPAPI_API_KEY=...`.
- La valeur par défaut `data_location = "data"` correspondra au point de montage `/app/data` utilisé ci-dessous.

### Étape 3 – Créer un dossier persistant sur l’hôte
```bash
mkdir -p ~/mywi_data
```
- Windows (PowerShell) : `New-Item -ItemType Directory -Path "C:/Users/vous/mywi_data"`

### Étape 4 – Construire l’image Docker
```bash
docker build -t mwi:latest .
```
- Pour inclure les dépendances ML (FAISS + transformers), exécuter `MYWI_WITH_ML=1 docker build -t mwi:latest .`.

### Étape 5 – Démarrer le conteneur
Remplacer le chemin hôte par le dossier créé à l’étape précédente.
```bash
docker run -dit --name mwi -v /chemin/vers/vos/donnees:/app/data mwi:latest
```
- macOS/Linux : `docker run -dit --name mwi -v ~/mywi_data:/app/data mwi:latest`
- Windows : `docker run -dit --name mwi -v C:/Users/vous/mywi_data:/app/data mwi:latest`
- Ajouter des variables d’environnement (`-e MWI_SERPAPI_API_KEY=...`) si vous ne souhaitez pas modifier `settings.py`.

### Étape 6 – Initialiser la base (première fois uniquement)
```bash
docker exec -it mwi python mywi.py db setup
```

### Étape 7 – Utiliser la CLI depuis le conteneur
```bash
docker exec -it mwi python mywi.py land list
```
Exécuter ensuite toutes les commandes sous la forme `docker exec -it mwi python mywi.py ...`. Tapez `exit` pour quitter un shell ouvert avec `docker exec -it mwi bash`.

### Gérer le conteneur
```bash
docker stop mwi         # arrêter le conteneur
docker start mwi        # le relancer plus tard
docker rm mwi           # supprimer le conteneur (les données restent sur l’hôte)
```

Vos données demeurent dans le dossier monté à l’étape 5. Relancer le conteneur avec la même option `-v` réutilise la base existante.

## Installation locale

1. Cloner le dépôt et créer un environnement virtuel :
   ```bash
   git clone https://github.com/MyWebIntelligence/MyWebIntelligencePython.git
   cd MyWebIntelligencePython
   python -m venv venv
   source venv/bin/activate  # Windows : .\venv\Scripts\activate
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Configurer `settings.py:data_location` vers un chemin absolu accessible.
4. Initialiser la base :
   ```bash
   python mywi.py db setup
   ```
5. (Optionnel) Installer Playwright pour l’analyse média dynamique :
   ```bash
   python install_playwright.py
   ```

---

## Notes générales

- Les commandes se lancent avec `python mywi.py ...` (ou, via Docker, `docker compose exec mwi python mywi.py ...`).
- Remplacez les valeurs d’exemple (`LAND`, `TERMS`, `https://…`) par vos données.
- Vérifiez que `settings.py:data_location` pointe vers un dossier existant et accessible en écriture.

---

## Gestion des lands

### 1. Créer un nouveau land
```bash
python mywi.py land create --name="MonSujet" --desc="Description" --lang=fr
```

### 2. Lister les lands
```bash
python mywi.py land list
python mywi.py land list --name="MonSujet"  # détail d’un land donné
```

### 3. Ajouter des termes
```bash
python mywi.py land addterm --land="MonSujet" --terms="mot1, mot2"
```

### 4. Ajouter des URLs
```bash
python mywi.py land addurl --land="MonSujet" --urls="https://example.org,https://example.net"
python mywi.py land addurl --land="MonSujet" --path=urls.txt
```

### 5. Récupérer des URLs via SerpAPI
```bash
python mywi.py land urlist --name="MonSujet" --query="(mot clé)" \
  --datestart=2023-01-01 --dateend=2023-03-31 --timestep=week --lang=fr
```
- Nécessite `settings.serpapi_api_key` ou `MWI_SERPAPI_API_KEY`.
- `--sleep` contrôle la pause entre pages (défaut : 1 s).
- Si une plage de dates est fournie (ou si vous ajoutez `--progress`), le terminal
  affiche une ligne par fenêtre temporelle avec les dates couvertes et le nombre
  d'URLs récupérées via SerpAPI.

### 6. Supprimer des données
- Supprimer un land entier :
  ```bash
  python mywi.py land delete --name="MonSujet"
  ```
- Supprimer les expressions sous un seuil de pertinence :
  ```bash
  python mywi.py land delete --name="MonSujet" --maxrel=0.5
  ```

---

## Collecte de données

### 1. Crawler les URLs du land
```bash
python mywi.py land crawl --name="MonSujet" [--limit N] [--http CODE] [--depth D]
```
- `--limit` : nombre maximal d’URLs à crawler.
- `--http` : relance uniquement les pages ayant renvoyé ce code (ex. `--http 503`).
- `--depth` : limite la profondeur de crawl.

### 2. Extraire un contenu lisible (pipeline Mercury)

Pré-requis : `npm install -g @postlight/mercury-parser`
```bash
python mywi.py land readable --name="MonSujet" [--limit N] [--depth D] [--merge stratégie]
```
Stratégies de fusion :
- `smart_merge` (défaut) : privilégie Mercury pour le contenu, conserve les métadonnées les plus riches.
- `mercury_priority` : Mercury écrase toujours les données existantes.
- `preserve_existing` : complète uniquement les champs vides.

### 3. Capturer les métriques SEO Rank
```bash
python mywi.py land seorank --name="MonSujet" --limit=100 --depth=1 --force
```
- Clé à renseigner dans `settings.seorank_api_key` ou `MWI_SEORANK_API_KEY`.
- Options : `--http`, `--minrel`, `--request-delay`.
- Par défaut, seules les expressions `http_status=200` et `relevance ≥ 1` sont enrichies.

---

## Export de données

```bash
python mywi.py land export --name="MonSujet" --type=pagecsv
python mywi.py land export --name="MonSujet" --type=nodegexf
python mywi.py land export --name="MonSujet" --type=mediacsv
python mywi.py land export --name="MonSujet" --type=corpus
```
Types disponibles : `pagecsv`, `pagegexf`, `nodecsv`, `nodegexf`, `mediacsv`, `corpus`, `pseudolinks`, `pseudolinkspage`, `pseudolinksdomain`.

Les exports sont déposés dans `data/export_*` sous le `data_location` configuré.

---

## Mettre à jour les domaines depuis les heuristiques

```bash
python mywi.py heuristic update
```
Met à jour les domaines en appliquant les heuristiques configurées (pertinence, filtres, métadonnées). Utile après modification de `settings.heuristics`.

---

## Pipeline de consolidation des lands

```bash
python mywi.py land consolidate --name="MonSujet" [--limit N] [--depth D]
```
- Reconstruit les liens médias/expressions manquants, recalcul les scores de pertinence, complète les métadonnées.
- Ne traite que les expressions déjà crawlées (`fetched_at` non nul).

---

## Tests

- Exécuter l’ensemble :
  ```bash
  pytest tests/
  ```
- Fichier spécifique :
  ```bash
  pytest tests/test_cli.py
  ```
- Test ciblé :
  ```bash
  pytest tests/test_cli.py::test_functional_test
  ```

---

# Embeddings & pseudolinks (guide utilisateur)

## Objectif
- Générer des vecteurs par paragraphe puis relier les paragraphes proches (pseudolinks).
- Facultatif : qualifier chaque paire avec un modèle NLI (entailment / neutral / contradiction).
- Exporter les relations au niveau paragraphe, page et domaine.

Flux type :
1. Crawler et extraire le texte lisible.
2. Générer les embeddings (`embedding generate`).
3. Calculer les similarités (`embedding similarity`).
4. Exporter les résultats (`land export --type=pseudolinks|pseudolinkspage|pseudolinksdomain`).

## Pré-requis & installation
- Base initialisée et pages avec contenu lisible.
- Créer un environnement virtuel Python « propre » :
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip setuptools wheel
  python -m pip install -r requirements.txt
  ```
- Optionnel (NLI + FAISS) :
  ```bash
  python -m pip install -r requirements-ml.txt
  ```
- Vérifier l’environnement :
  ```bash
  python mywi.py embedding check
  ```

## Modèles
- Modèle NLI multilingue recommandé : `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
- Fallback léger (anglais) : `typeform/distilbert-base-uncased-mnli`.
- Configurer `settings.py:nli_model_name` (et `nli_fallback_model_name` si souhaité).

## Paramètres (référence)
- **Embeddings (bi-encoder)** :
  - `embed_provider`, `embed_model_name`, `embed_batch_size`, `embed_min_paragraph_chars`, `embed_max_paragraph_chars`.
  - `embed_similarity_method` (`cosine`, `cosine_lsh`), `embed_similarity_threshold`.
- **Rappel ANN / NLI** :
  - `similarity_backend` (`faiss` | `bruteforce`).
  - `similarity_top_k` (voisins par paragraphe).
  - `nli_model_name`, `nli_fallback_model_name`.
  - `nli_backend_preference`, `nli_batch_size`, `nli_max_tokens`, `nli_torch_num_threads`.
  - `nli_progress_every_pairs`, `nli_show_throughput`.
- **Variables d’environnement CPU** :
  - `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TOKENIZERS_PARALLELISM=false`.

## Commandes & paramètres
- Générer les embeddings :
  ```bash
  python mywi.py embedding generate --name=LAND [--limit N]
  ```
- Calculer les similarités :
  - Cosine exacte :
    ```bash
    python mywi.py embedding similarity --name=LAND --method=cosine \
      --threshold=0.85 [--minrel R]
    ```
  - Cosine LSH (approx) :
    ```bash
    python mywi.py embedding similarity --name=LAND --method=cosine_lsh \
      --lshbits=20 --topk=15 --threshold=0.85 [--minrel R] [--maxpairs M]
    ```
  - ANN + NLI :
    ```bash
    python mywi.py embedding similarity --name=LAND --method=nli \
      --backend=faiss|bruteforce --topk=10 [--minrel R] [--maxpairs M]
    ```
- Exports CSV :
  ```bash
  python mywi.py land export --name=LAND --type=pseudolinks
  python mywi.py land export --name=LAND --type=pseudolinkspage
  python mywi.py land export --name=LAND --type=pseudolinksdomain
  ```
- Outils :
  - Vérifier la configuration : `python mywi.py embedding check`
  - Réinitialiser un land : `python mywi.py embedding reset --name=LAND`

## Dépannage & précautions
- Scores `score_raw=0.5` et `score=0` → le modèle de secours neutre a été utilisé. Installer les dépendances ML ou choisir un modèle supporté.
- Colonne absente (`score_raw`) → lancer `python mywi.py db migrate`.
- Segfault macOS (OpenMP/Torch) → venv pip-only, commencer avec `OMP_NUM_THREADS=1`, ajuster ensuite ; éventuellement `KMP_DUPLICATE_LIB_OK=TRUE`.
- Scoring lent → diminuer `nli_batch_size`, filtrer avec `--minrel`, limiter avec `--maxpairs`, ajuster les threads.
- Trop de paires → augmenter `--threshold`, `--lshbits`, réduire `--topk`, appliquer `--minrel`.

## Bonnes pratiques — performance
- **Petit/moyen corpus (≤ ~50k paragraphes)** : méthode `cosine`, `--threshold=0.85`, `--minrel=1`.
- **Grand corpus (≥ ~100k)** : privilégier `cosine_lsh`, régler `--lshbits=18–22`, `--topk=10–20`, `--threshold` ≥ 0,85, limiter `--maxpairs`.
- **Pipeline NLI** :
  - Utiliser FAISS (`--backend=faiss`) si disponible.
  - Départ prudent : `--topk=6–10`, `--minrel=1–2`, `--maxpairs=20k–200k`.
  - Choisir un modèle adapté (DistilBERT MNLI pour des tests rapides, DeBERTa XNLI pour qualité multilingue).
  - Ajuster `nli_batch_size` (32–96) et `nli_max_tokens` (384–512).
- **Threads CPU** :
  - Exporter `OMP_NUM_THREADS=N` et caler `settings.nli_torch_num_threads` sur la même valeur.
  - Garder `TOKENIZERS_PARALLELISM=false`.
- **Suivi** : `nli_progress_every_pairs` contrôle la fréquence d’affichage (progression, débit, ETA).

## Choix des modèles et recours
- Modèle NLI par défaut : DeBERTa multilingue (requiert `sentencepiece`).
- Alternative sûre (EN) : DistilBERT MNLI.
- Configurer dans `settings.py:nli_model_name`.
- En absence de dépendances, le système retombe sur un classifieur neutre (`score=0`).

## Progression & logs
- Le rappel ANN journalise le nombre de candidats tous les quelques centaines de paragraphes.
- Le scoring NLI affiche périodiquement `pairs/s`, ETA et cumul.
- Un résumé final présente le temps total et le volume traité.

## Méthodes de similarité
- `cosine` : comparaison exacte O(n²), idéale pour volumes modestes.
- `cosine_lsh` : recherche approximative via hyperplans aléatoires, scalable sans FAISS.
- `nli` : pipeline en deux étapes (ANN → cross-encoder NLI) produisant `RelationScore` { -1, 0, 1 } et `ConfidenceScore`.

## Choisir le backend ANN (FAISS)
- Installer FAISS (`pip install faiss-cpu`) pour accélérer le rappel.
- Forcer un backend : `--backend=faiss` (ou `--backend=bruteforce`).
- Paramètre global : `similarity_backend = 'faiss'` dans `settings.py`.
- Sans FAISS, le rappel repasse automatiquement en bruteforce.
- Vérification : `python mywi.py embedding check`.

## Similarité scalable (lands volumineux)

```bash
python mywi.py embedding similarity \
  --name=MonSujet \
  --method=cosine_lsh \
  --threshold=0.85 \
  --lshbits=20 \
  --topk=15 \
  --minrel=1 \
  --maxpairs=5000000
```

- `--lshbits` : nombre d’hyperplans (plus élevé → buckets plus fins).
- `--topk` : limite le nombre de voisins conservés par paragraphe.
- `--threshold` : seuil cosine minimal.
- `--minrel` : filtre sur la pertinence des expressions.
- `--maxpairs` : plafond sur le nombre total de paires.

## Relations NLI (ANN + cross-encoder)

- Installer les dépendances si besoin :
  ```bash
  pip install sentence-transformers transformers
  pip install faiss-cpu  # rappel ANN plus rapide (optionnel)
  ```
- Exemple :
  ```bash
  python mywi.py embedding similarity \
    --name=MonSujet \
    --method=nli \
    --backend=bruteforce \
    --topk=50 \
    --minrel=1 \
    --maxpairs=2000000
  ```
- Réglages clés : `nli_model_name`, `nli_batch_size`, `similarity_backend`, `similarity_top_k`.
- Recettes rapides :
  - Cosine exact : `--method=cosine --threshold=0.85 --minrel=1`.
  - Cosine approx : `--method=cosine_lsh --lshbits=20 --topk=15 --threshold=0.85 --minrel=1 --maxpairs=5000000`.
  - ANN + NLI + FAISS :
    ```bash
    pip install sentence-transformers transformers faiss-cpu
    python mywi.py embedding similarity --name=MonSujet --method=nli \
      --backend=faiss --topk=50 --minrel=1 --maxpairs=2000000
    ```
- Export CSV (pseudolinks) : `land export --type=pseudolinks|pseudolinkspage|pseudolinksdomain`.
- `python mywi.py embedding check` récapitule la configuration, les librairies détectées et l’état des tables.

---

# Dépannage & réparation

## Récupération SQLite

Si la base SQLite est corrompue (ex. « database disk image is malformed »), utilisez le script dédié :
```bash
chmod +x scripts/sqlite_recover.sh
scripts/sqlite_recover.sh data/mwi.db data/mwi_repaired.db
```

Étapes :
- Sauvegarde `data/mwi.db` (+ fichiers `-wal` / `-shm` le cas échéant) dans `data/sqlite_repair_<timestamp>/backup/`.
- Tente d’abord `.recover`, puis `.dump` en repli.
- Reconstruit `data/mwi_repaired.db`, exécute `PRAGMA integrity_check;`, journalise les tables dans `.../logs/`.

Valider sans écraser la base d’origine :
```bash
mkdir -p data/test-repaired
cp data/mwi_repaired.db data/test-repaired/mwi.db
MWI_DATA_LOCATION="$PWD/data/test-repaired" venv/bin/python mywi.py land list
```

Adopter la base réparée (après sauvegarde manuelle) :
```bash
cp data/mwi.db data/mwi.db.bak_$(date +%Y%m%d_%H%M%S)
mv data/mwi_repaired.db data/mwi.db
```

Astuce : `MWI_DATA_LOCATION` permet de pointer temporairement vers un autre dossier de données sans modifier `settings.py:data_location`.

---

# Licence

MyWI est distribué sous licence MIT. Consultez le fichier [LICENSE](LICENSE) pour le détail des conditions.
