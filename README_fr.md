# My Web Intelligence (MyWI)

Version anglaise: [README.md](README.md)

MyWebIntelligence (MyWI) est un outil Python pour aider les chercheur·e·s en humanités numériques à créer et piloter des projets d’exploration du Web. Il facilite la collecte, l’organisation et l’analyse de données web, en stockant le tout dans une base SQLite. Pour consulter la base, un outil comme [SQLiteBrowser](https://sqlitebrowser.org/) est pratique.

## Sommaire

- Fonctionnalités
- Architecture & internes
- Installation (Docker Compose recommandé, Docker manuel, Local)
- Utilisation (Lands, Crawl, Readable, Domaines, Exports, Heuristiques)
- Tests
- Embeddings & Pseudolinks
- Récupération SQLite
- Licence

## Fonctionnalités

- Land (projet) : regrouper termes et URL par thème.
- Crawl : récupérer HTML et métadonnées des pages.
- Readable : extraire un contenu lisible de haute qualité (Mercury autonome).
- Embeddings & pseudolinks : embeddings par paragraphe, similarité sémantique, CSV des « pseudolinks » entre paragraphes proches.
- Analyse média : extraction/inspection des images/vidéos/audio (dimensions, taille, format, couleurs dominantes, EXIF), filtres, détection doublons, traitement asynchrone par lots.
- Détection médias améliorée : extensions majuscules/minuscules supportées (.JPG/.jpg, .PNG/.png, …).
- Extraction médias dynamique (optionnelle) : navigateur headless pour contenus JS et lazy-loading (Playwright).
- Analyse domaine : collecte d’infos sur les domaines rencontrés.
- Exports : CSV, GEXF, corpus brut, pseudolinks paragraphe/page/domaine.
- Tags : export matrices et contenus tagués.

---

## Architecture & internes

```
mywi.py  →  mwi/cli.py  →  mwi/controller.py  →  mwi/core.py & mwi/export.py
                                     ↘︎ mwi/model.py (Peewee ORM)
                                     ↘︎ mwi/embedding_pipeline.py (embeddings par paragraphe)
```

- `mywi.py` : point d’entrée CLI.
- `mwi/cli.py` : parsing des arguments et dispatch vers les contrôleurs.
- `mwi/controller.py` : mapping des verbes vers la logique core/export/model.
- `mwi/core.py` : algorithmes principaux (crawl, parsing, pipelines, scoring, etc.).
- `mwi/export.py` : exports (CSV, GEXF, corpus).
- `mwi/model.py` : schéma DB (Peewee ORM).

### Schéma (SQLite, via Peewee)
- Land, Word, LandDictionary (M2M), Domain, Expression, ExpressionLink
- Media (images/vidéos/audio)
- Paragraph / ParagraphEmbedding / ParagraphSimilarity (pseudolinks)
- Tag, TaggedContent

### Workflows clés
- Bootstrap DB : `python mywi.py db setup`
- Cycle Land : créer → termes → URLs → crawl → readable → exports → nettoyage
- Domaines : `python mywi.py domain crawl`
- Tags : `python mywi.py tag export`
- Heuristiques : `python mywi.py heuristic update`
- Embeddings & similarité :
  - Générer : `python mywi.py embedding generate --name=LAND [--limit N]`
  - Similarité : `python mywi.py embedding similarity --name=LAND --method=cosine|cosine_lsh|nli`

### Notes d’implémentation
- Pertinence : somme pondérée des lemmes trouvés (titre/contenu).
- Concurrence polie : batching async pour le crawl.
- Médias : extraction et liaison, conservation des `.jpg` pour DL ultérieur (selon config).
- Export : formats multiples, SQL dynamique, GEXF avec attributs.

### Réglages (settings.py)
- `data_location`, `user_agent`, `parallel_connections`, `default_timeout`, `archive`, `heuristics`.

#### Embeddings — configuration
- `embed_provider` : `fake` | `http` | `openai` | `mistral` | `gemini` | `huggingface` | `ollama`
- `embed_api_url` : URL du provider HTTP générique (POST `{model, input}`)
- `embed_model_name` : libellé du modèle stocké avec les vecteurs
- `embed_batch_size` : taille de lot
- `embed_min_paragraph_chars` / `embed_max_paragraph_chars` : bornes de longueur des paragraphes
- `embed_similarity_method` / `embed_similarity_threshold`

Clés spécifiques :
- OpenAI/Mistral : payload `{ "model": name, "input": [texts...] }` → `{ "data": [{"embedding": [...]}, ...] }`
- Gemini : `:batchEmbedContents` → `{ "embeddings": [{"values": [...]}, ...] }`
- Hugging Face : `{ "inputs": [texts...] }` → liste de vecteurs
- Ollama (local) : pas de batch, appels séquentiels à `/api/embeddings` avec `{ "model": name, "prompt": text }`

#### Optionnel : Filtre de pertinence OpenRouter (LLM oui/non)
- Variables : `MWI_OPENROUTER_ENABLED`, `MWI_OPENROUTER_API_KEY`, `MWI_OPENROUTER_MODEL`, `MWI_OPENROUTER_TIMEOUT`, `MWI_OPENROUTER_READABLE_MAX_CHARS`, `MWI_OPENROUTER_MAX_CALLS_PER_RUN`.
- Désactivé → comportement identique à l’existant.

---

## Installation

Vous pouvez utiliser Docker Compose (recommandé) ou installer localement.

### Docker Compose (recommandé)

Pré-requis : Docker Desktop/Engine + Compose.

0) Optionnel — choisir l’emplacement des données/exports
- Copier `.env.example` en `.env` puis laisser par défaut : `HOST_DATA_DIR=./data` (dans le dépôt), ou bien définir un chemin absolu (ex. macOS/Linux: `/Users/vous/mywi_data`, Windows: `C:/Users/Vous/mywi_data`).

1) Construire et démarrer
```bash
docker compose up -d --build
```

2) Initialiser la base (première fois)
```bash
docker compose exec mwi python mywi.py db setup
```

3) Exemples CLI
```bash
docker compose exec mwi python mywi.py land create --name="MonSujet" --desc="…" --lang=fr
docker compose exec mwi python mywi.py land addurl --land="MonSujet" --urls="https://example.org"
docker compose exec mwi python mywi.py land crawl --name="MonSujet" --limit=10
docker compose exec mwi python mywi.py land readable --name="MonSujet" --merge=smart_merge
docker compose exec mwi python mywi.py land export --name="MonSujet" --type=pagecsv
```

Où sont les fichiers ?
- Hôte : `${HOST_DATA_DIR}` (défaut `./data` dans le dépôt)
- Conteneur : `/app/data`
- `settings.py` utilise déjà `data` (→ `/app/data`).

Optionnels :
- Installer les navigateurs Playwright (extraction dynamique médias) :
```bash
docker compose exec mwi python install_playwright.py
```
- Construire avec extras ML (FAISS + transformers) :
```bash
MYWI_WITH_ML=1 docker compose build
# Puis `docker compose up -d`
```

Arrêt / suppression :
```bash
docker compose down        # stop
docker compose down -v     # stop + volumes (destructif)
```

Choix d’emplacement des données — cas courants
- Cas A (défaut, simple) : `HOST_DATA_DIR=./data` dans `.env`.
- Cas B (hors dépôt) : chemin absolu dans `.env`, puis `docker compose up -d --build`.

### Docker (manuel)

1) Créer un dossier de données (hôte), p.ex. `~/mywi_data`.

2) Cloner le projet puis `cd MyWebIntelligencePython`.

3) Monter le dossier hôte :
- Cas recommandé : vers `/app/data` (aucune modification d’app).
- Cas avancé : autre chemin interne + `-e MYWI_DATA_DIR=/data`.

4) Builder l’image : `docker build -t mwi:latest .`

5) Lancer : `docker run -dit --name mwi -v /chemin/hote:/app/data mwi:latest`

6) Shell conteneur : `docker exec -it mwi bash`

7) Init DB (dans le conteneur) : `python mywi.py db setup`

Remarque Playwright : l’image inclut Playwright+Chromium pour médias dynamiques.

### Installation locale (développement)

Pré-requis : Python 3.10+, pip, virtualenv, git.

1) Créer/activer un venv
2) Configurer `settings.py:data_location` vers un dossier absolu
3) Installer deps : `pip install -r requirements.txt`
4) Optionnel Playwright : `python install_playwright.py`
5) Init DB : `python mywi.py db setup`

---

## Utilisation

### Notes générales
- Exécuter via `python mywi.py …` (ou `docker compose exec mwi python mywi.py …`).
- Remplacer les placeholders (`LAND`, `TERMS`, etc.).

### Lands — pipeline consolidation
- Répare/reconstruit la cohérence d’un land (pertinence, liens, médias) :
```bash
python mywi.py land consolidate --name=LAND [--limit N] [--depth D]
```
- Traite uniquement les pages déjà crawlées (`fetched_at` non null).

### Créer / lister / enrichir un Land
- Créer :
```bash
python mywi.py land create --name="MonSujet" --desc="…" --lang=fr|en
```
- Lister : `python mywi.py land list` (ou `--name` pour le détail)
- Ajouter des termes :
```bash
python mywi.py land addterm --land="MonSujet" --terms="mot1, mot2"
```
- Ajouter des URLs (directement ou via fichier) :
```bash
python mywi.py land addurl --land="MonSujet" --urls="https://…"
python mywi.py land addurl --land="MonSujet" --path="/chemin/urls.txt"
```
- Supprimer land ou expressions sous un seuil :
```bash
python mywi.py land delete --name="MonSujet" [--maxrel=0.5]
```

### Collecte (crawl)
```bash
python mywi.py land crawl --name="MonSujet" [--limit N] [--http 503] [--depth D]
```

### Extraction lisible (Mercury autonome)
Pré-requis : `npm i -g @postlight/mercury-parser`
```bash
python mywi.py land readable --name="MonSujet" [--limit N] [--depth D] [--merge smart_merge|mercury_priority|preserve_existing]
```
Caractéristiques : extraction propre, fusion intelligente, enrichissement auto (médias, liens), recalcul pertinence.

### Domaines
```bash
python mywi.py domain crawl [--limit N] [--http 404]
```

### Exports
```bash
python mywi.py land export --name="MonSujet" --type=pagecsv|pagegexf|fullpagecsv|nodecsv|nodegexf|mediacsv|corpus|pseudolinks|pseudolinkspage|pseudolinksdomain [--minrel R]
```

### Médias — analyse
```bash
python mywi.py land medianalyse --name=LAND [--depth D] [--minrel R]
```
Résultats en base (dimensions, taille, format, couleurs, EXIF, pHash). Réglages dans `settings.py`.

### Heuristiques
```bash
python mywi.py heuristic update
```

---

## Tests

Exécuter tous les tests : `pytest tests/`
Exécuter un test ciblé : `pytest tests/test_cli.py::test_functional_test`

---

## Embeddings & Pseudolinks (guide rapide)

Objectif : créer des vecteurs par paragraphe, relier les plus proches (pseudolinks), éventuellement classifier la relation via NLI (entail/neutral/contradict). Exports au niveau paragraphe, page et domaine.

Pré-requis : DB initialisée et textes lisibles présents. En venv, installer `requirements.txt`, puis optionnellement `requirements-ml.txt` (NLI + FAISS). Vérifier : `python mywi.py embedding check`.

Commandes :
- Générer :
```bash
python mywi.py embedding generate --name=LAND [--limit N]
```
- Similarité (au choix) :
```bash
python mywi.py embedding similarity --name=LAND --method=cosine --threshold=0.85 [--minrel R]
python mywi.py embedding similarity --name=LAND --method=cosine_lsh --lshbits=20 --topk=15 --threshold=0.85 [--minrel R] [--maxpairs M]
python mywi.py embedding similarity --name=LAND --method=nli --backend=faiss|bruteforce --topk=10 [--minrel R] [--maxpairs M]
```
- Exports :
```bash
python mywi.py land export --name=LAND --type=pseudolinks
python mywi.py land export --name=LAND --type=pseudolinkspage
python mywi.py land export --name=LAND --type=pseudolinksdomain
```

Réglages utiles :
- Embeddings : `embed_provider`, `embed_model_name`, `embed_batch_size`, `embed_min_paragraph_chars`, `embed_max_paragraph_chars`, `embed_similarity_method`, `embed_similarity_threshold`.
- ANN/NLI : `similarity_backend` (`faiss`|`bruteforce`), `similarity_top_k`, `nli_model_name`, `nli_fallback_model_name`, `nli_backend_preference`, `nli_batch_size`, `nli_max_tokens`, `nli_torch_num_threads`.
- CPU : exporter `OMP_NUM_THREADS=N` (et selon besoin `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TOKENIZERS_PARALLELISM=false`).

Bonnes pratiques performance :
- Petit/moyen (≤ ~50k paragraphes) : méthode exacte `cosine` avec `--threshold=0.85`, `--minrel=1`.
- Grand (≥ ~100k) : préférer `cosine_lsh` (approx) et borner fan‑out + sortie (`--lshbits`, `--topk`, `--threshold`, `--minrel`, `--maxpairs`).
- NLI : commencer petit (`--topk=6–10`, `--minrel=1–2`, `--maxpairs=20k–200k`), choisir le modèle (ex. EN rapide: `typeform/distilbert-base-uncased-mnli`; multilingue: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`), ajuster `nli_batch_size` et `nli_max_tokens`.

---

## Récupération SQLite

Si la base SQLite est corrompue (ex. « database disk image is malformed »), utiliser le script d’aide :
```bash
chmod +x scripts/sqlite_recover.sh
scripts/sqlite_recover.sh data/mwi.db data/mwi_repaired.db
```
Il sauvegarde, tente `.recover` puis `.dump`, reconstruit une nouvelle DB et vérifie l’intégrité. Tester ensuite via `MWI_DATA_LOCATION=... python mywi.py land list` sans écraser l’original.

## Licence

Projet sous la licence indiquée dans le fichier LICENSE.

