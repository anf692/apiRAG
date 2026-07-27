# RAG Règlement Intérieur API Django

API Django (Django REST Framework) qui permet de poser des questions sur un document de règlement intérieur (PDF) en utilisant une architecture RAG (Retrieval-Augmented Generation), avec un second LLM qui évalue automatiquement la fiabilité de chaque réponse (LLM-as-a-judge / groundedness).

## Fonctionnement

1. Le PDF (`reglements.pdf`) est découpé en chunks et vectorisé une seule fois dans une base Chroma persistante.
2. À chaque question reçue via l'API, les chunks les plus pertinents sont récupérés (`retriever`).
3. Un premier LLM (`openai/gpt-oss-20b:free` via OpenRouter) génère une réponse **strictement basée sur le contexte récupéré**. Si l'information n'est pas dans le document, il répond `JE NE SAIS PAS`.
4. Un second LLM (`nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter) joue le rôle de juge et évalue si la réponse est bien fondée sur le contexte (score de 1 à 5 + analyse détaillée).

## Stack technique

- **Backend** : Django + Django REST Framework
- **Documentation API** : drf-spectacular
- **RAG** : LangChain
- **Base vectorielle** : Chroma (persistante sur disque)
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace, local)
- **LLMs** : OpenRouter (compatible API OpenAI)

## Prérequis

- Python 3.10+
- Un compte [OpenRouter](https://openrouter.ai/) avec une clé API

## Installation

```bash
# 1. Cloner le projet et se placer dedans
cd ton_projet_django

# 2. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install django djangorestframework drf-spectacular
pip install langchain langchain-text-splitters langchain-community
pip install langchain-chroma langchain-huggingface langchain-openai
pip install pypdf sentence-transformers python-dotenv
```

## Configuration

Crée un fichier `.env` à la racine du projet (au même niveau que `manage.py`) :

```env
OPENROUTER_API_KEY=ta_cle_openrouter_ici
```

Place ton PDF de règlement intérieur ici :

```
app/services/pdfs/reglements.pdf
```

> Adapte le chemin `PDF_FILE` dans `rag_pipeline.py` si ton PDF est ailleurs.


## Lancement

```bash
python manage.py runserver
```

Au premier appel, si `db_vector/` n'existe pas encore, le PDF est chargé, découpé et vectorisé (peut prendre quelques secondes à quelques minutes selon la taille du document). Les appels suivants réutilisent la base déjà persistée.

## Utilisation de l'API

### Endpoint

```
POST /rag/
```

### Corps de la requête

```json
{
  "question": "Quelles sont les règles concernant les retards ?"
}
```

### Exemple avec curl

```bash
curl -X POST http://127.0.0.1:8000/rag/ \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelles sont les règles concernant les retards ?"}'
```

### Réponse

```json
{
  "question": "Quelles sont les règles concernant les retards ?",
  "answer": "Selon l'Article 12, tout retard de plus de 15 minutes doit être justifié...",
  "evaluation": "Analyse:\n- ...\n\nPoints corrects:\n- ...\n\nErreurs / hallucinations:\n- ...\n\nScore: 5/5",
  "confidence": "high"
}
```

- `confidence: "low"` → la réponse générée contient `JE NE SAIS PAS` (information absente du document).
- `confidence: "high"` → une réponse a été trouvée dans le contexte.
- `evaluation` → analyse détaillée du second LLM sur la fiabilité de la réponse (groundedness).

### Documentation interactive (Swagger / drf-spectacular)

Si `drf-spectacular` est branché dans `urls.py` du projet principal, la doc est disponible sur :

```
/api/schema/swagger-ui/
```

## Points d'attention

- Les modèles utilisés (`openai/gpt-oss-20b:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`) sont des modèles **gratuits** sur OpenRouter : ils peuvent avoir des limites de débit (rate limit) ou être temporairement indisponibles.
- La base vectorielle (`db_vector/`) n'est reconstruite que si le dossier n'existe pas ou est vide. Pour forcer une reconstruction (ex: après mise à jour du PDF), supprime le dossier `db_vector/` avant de relancer le serveur.
- Le pipeline (chargement du PDF + embeddings) est actuellement exécuté au niveau du module, donc dès l'import de `rag_pipeline.py` par Django.
