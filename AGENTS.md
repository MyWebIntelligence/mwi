# MyWebIntelligence – Guide AGENTS pour Vibe Coding

Ce document sert de point d’entrée pour toute session de vibe coding avec un agent (Claude, ChatGPT, etc.) sur ce repo. Il condense l’essentiel pour être opérationnel rapidement, cadrer la collaboration, et éviter les écueils.

**Sources de vérité**
- `.claude/CLAUDE.md:1`
- `memory-bank/Agent.md:1`
- `memory-bank/Pipelines.md:1`
- `README.md:1`

## Objectif

- **But**: livrer vite des contributions sûres et ciblées (CLI, pipelines, exports), sans régressions.
- **Approche**: clarifier → planifier → exécuter par patchs minimaux → valider.
- **Style**: concis, pragmatique, aligné sur l’existant (Peewee ORM, contrôleurs fins, CLI mwi).

## Kickoff Session

- **Contexte minimum**: objectif, jeu d’essai (land/URLs), environnement (Docker/local), état DB.
- **Checklist**:
  - DB prête (`python mywi.py db setup`).
  - `settings.py:data_location` pointe vers un dossier accessible.
  - Dépendances installées (`pip install -r requirements.txt`). Optionnel: `python install_playwright.py`.
  - Accès aux commandes `python mywi.py …` OK.
  - Clé SEO Rank renseignée si besoin (`settings.seorank_api_key` ou `MWI_SEORANK_API_KEY`).
  - Clé SerpAPI renseignée pour `land urlist` (`settings.serpapi_api_key` ou `MWI_SERPAPI_API_KEY`).
- **Plan de travail**:
  - Lire rapidement `README.md`, `memory-bank/Agent.md`, `memory-bank/Pipelines.md`.
  - Lister l’impact (CLI, core, export, model, settings).
  - Proposer un plan en 3–5 étapes vérifiables, puis exécuter par patchs.

## Repo Map Essentielle

- **Entrées**: `mywi.py:1` (CLI), `mwi/cli.py:1` (parsing/dispatch), `mwi/controller.py:1` (façade).
- **Cœur**: `mwi/core.py:1` (crawlers, readable, heuristics, scoring), `mwi/export.py:1` (CSV/GEXF/Corpus), `mwi/model.py:1` (Peewee).
- **Pipelines**: `mwi/readable_pipeline.py:1` (Mercury autonome), `mwi/media_analyzer.py:1` (analyse média).
- **Config**: `settings.py:1` (paths, timeouts, parallélisme, heuristics, médias).
- **Tests**: `tests/:1` (CLI, core, extraction, exports).

## Démarrages Rapides

- **Docker**:
  - Construire: `docker build -t mwi:latest .`
  - Lancer: `docker run -dit --name mwi -v /path/host/data:/data mwi:latest`
  - Shell: `docker exec -it mwi bash`
  - Init DB: `python mywi.py db setup`
- **Local (venv)**:
  - Créer/activer venv, installer deps: `pip install -r requirements.txt`
  - Configurer `settings.py:data_location`
  - Init DB: `python mywi.py db setup`

## Cheatsheet CLI (jour 1)

- **Lands**:
  - Créer: `python mywi.py land create --name=LAND --desc="…" --lang=fr|en`
  - Termes: `python mywi.py land addterm --land=LAND --terms="k1, k2"`
  - URLs: `python mywi.py land addurl --land=LAND --urls="https://…"` ou `--path=file.txt`
  - SERP → URLs: `python mywi.py land urlist --name=LAND --query="mot clé" [--datestart=AAAA-MM-JJ --dateend=AAAA-MM-JJ --timestep=week]`
    - Utilise SerpAPI pour récupérer les résultats Google et insérer les URLs absentes dans le land.
    - Nécessite `settings.serpapi_api_key` (ou `MWI_SERPAPI_API_KEY`).
  - Crawl: `python mywi.py land crawl --name=LAND [--limit N] [--depth D]`
  - Readable (Mercury): `python mywi.py land readable --name=LAND --merge=smart_merge`
  - SEO Rank: `python mywi.py land seorank --name=LAND [--limit N] [--depth D] [--force]` (nécessite `settings.seorank_api_key` ou `MWI_SEORANK_API_KEY`)
  - LLM validate (OpenRouter, oui/non): `python mywi.py land llm validate --name=LAND [--limit N] [--force]`
  - Consolidate: `python mywi.py land consolidate --name=LAND [--limit N] [--depth D]`
  - Export: `python mywi.py land export --name=LAND --type=pagecsv|pagegexf|nodecsv|nodegexf|mediacsv|corpus`
  - Supprimer: `python mywi.py land delete --name=LAND`
- **Médias**:
  - Analyse: `python mywi.py land medianalyse --name=LAND [--depth D] [--minrel R]`
  - Outils avancés: `python mywi.py land reanalyze|preview_deletion|media_stats …`
- **Domaines/Tags/Heuristics**:
  - Domain crawl: `python mywi.py domain crawl [--limit N] [--http 404]`
  - Tags export: `python mywi.py tag export --name=LAND --type=matrix|content`
  - Heuristics: `python mywi.py heuristic update`

### Embeddings & NLI (rapide)
- Générer embeddings: `python mywi.py embedding generate --name=LAND [--limit N]`
- Similarité scalable (cosine LSH):
  ```bash
  python mywi.py embedding similarity \
    --name=LAND --method=cosine_lsh \
    --threshold=0.85 --lshbits=20 --topk=15 --minrel=1 --maxpairs=5000000
  ```
- Relations logiques (ANN + NLI):
  ```bash
  # Optional: pip install sentence-transformers transformers faiss-cpu
  python mywi.py embedding similarity \
    --name=LAND --method=nli --backend=bruteforce --topk=50 --minrel=1 --maxpairs=2000000
  ```
- Export pseudolinks: `python mywi.py land export --name=LAND --type=pseudolinks`

## Pipelines – Quand utiliser quoi

- **Collecte initiale**: `land crawl`
  - Rapide, large échelle, HTML + métadonnées de base, liens.
- **Pré-seed SERP**: `land urlist`
  - Interroge Google via SerpAPI et ajoute uniquement les URLs absentes dans le land.
  - Supporte un filtrage temporel (`--datestart`, `--dateend`, `--timestep`).
  - Idéal pour construire une base initiale de domaines avant le crawl.
- **Extraction haute qualité**: `land readable` (Mercury autonome)
  - Markdown propre, liens/médias extraits, stratégies `smart_merge|mercury_priority|preserve_existing`.
- **Analyse média**: `land medianalyse` (+ commandes `land *media*`)
  - Métadonnées images/vidéos/audio, couleurs, EXIF, hash, filtres.
- **Réparation/Resync**: `land consolidate`
  - Reconstruit liens/médias, recalcul pertinence, ajoute manquants.
- **Enrichissement SEO**: `land seorank`
  - Récupère les métriques SEO Rank par URL et les stocke brutes dans `expression.seorank` (JSON).
  - Options utiles: `--limit` pour batch court, `--depth` pour filtrer, `--force` pour rafraîchir toutes les entrées.
  - Pré-requis: renseigner `settings.seorank_api_key` (ou variable env `MWI_SEORANK_API_KEY`).

## Règles d’Implémentation (agents)

- **Ciblage**: patchs minimaux, ne corrigez pas l’inhérent non lié.
- **Contrôleurs**: chaque verbe retourne `1` (succès) ou `0` (échec).
- **Style**: respecter patterns Peewee, exporters centralisés, logique métier dans `core.py`.
- **Paramétrage**: privilégier `settings.py` pour constantes (parallélisme, timeouts, heuristics, médias).
- **I/O**: pas d’IO sauvage; passer par couches existantes (controller → core/export/model).
- **Perfs**: batch async (`settings.parallel_connections`), timeouts, retries raisonnables.

## Qualité & Validation

- **Tests**:
  - Global: `pytest tests/`
  - Ciblé: `pytest tests/test_cli.py::test_functional_test`
  - Ajouter des tests seulement si logique nouvelle → suivre patterns existants.
- **Smoke local**:
  - Création land → 1–2 URLs → `crawl` → `readable` → export CSV.
  - Si médias: lancer une passe `land medianalyse` avec `--depth` court.
- **Exports**:
  - Sorties sous `data/export_*`. Vérifier nb de lignes/colonnes attendues.

## Collaboration Agent ↔ Humain

- **Clarifications d’entrée**:
  - Jeu d’essai (nom du land, URLs, profondeur, seuil pertinence).
  - Contrainte d’environnement (Docker/local, données volumineuses, réseau).
- **Preambles & Plans**:
  - Annoncer brièvement les actions (lecture code, patch, tests).
  - Maintenir un plan concis (3–5 étapes), maj à chaque phase.
- **Livraison**:
  - Fournir chemins de fichiers modifiés et commandes de validation.
  - Documenter rapidement l’impact et le rollback si besoin.

## Dépannage Express

- **DB absente/cassée**: `python mywi.py db setup` (destructif) ou vérifier `settings.data_location`.
- **Aucune page lisible**: tenter `land readable --merge=mercury_priority` ou `consolidate`.
- **Médias vides**: activer Playwright (`python install_playwright.py`), vérifier extensions et filtres `settings.py`.
- **Exports vides**: vérifier `--minrel`, profondeur, état `fetched_at/readable_at`.
- **Timeouts/HTTP**: baisser `parallel_connections`, augmenter `default_timeout`, tester `--http`.

## Modèles d’Interactions (exemples)

- **Kickoff**:
  - « Objectif: produire un export `pagecsv` propre pour LAND=X. DB initialisée, 10 URLs seed. On enchaîne: crawl → readable (smart_merge) → export. »
- **Patch court**:
  - « J’ajoute un nouveau type d’export CSV minimal (colonnes X,Y). Contrôleur + `export.py` + test CLI. »
- **Investigation**:
  - « Je trace le calcul de pertinence pour 3 pages qui sortent à 0 malgré bons termes. »

## Rappels Clés

- **Sécurité**: pas d’effacement de données sans demande explicite; annoncer toute action destructive.
- **Traçabilité**: référencer les fichiers modifiés; conserver la cohérence des couches.
- **Langue**: par défaut `fr`; passer `--lang` à la création du land si besoin.

---

Bonnes vibes et contributions utiles avant tout. 🚀
