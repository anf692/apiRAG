from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


# --- Configuration---

# charge le .env et si la variable existe deja dans le systeme ecrasse-la  avec celle du .env
load_dotenv(override=True) 
 
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY manquante. Vérifie ton fichier .env")



PDF_FILE = "./pdfs/reglements.pdf"
PERSIST_DIR = "./db_vector"
COLLECTION_NAME = "reglements_v1"
 

# --- LLM principal (répond aux questions) ---
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model="openai/gpt-oss-20b:free"
)


# --- 1. Chargement et découpage du PDF ---
loader = PyPDFLoader(PDF_FILE) #charge le pdf

#
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="o200k_base",
    chunk_size=300,
    chunk_overlap=50  # évite de couper une idée en plein milieu entre deux chunks
)

 
chunks = loader.load_and_split(text_splitter)
print(f"Nombre de chunks générés : {len(chunks)}")


# --- 2. Embeddings + base vectorielle ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

 
# On évite de reconstruire la base si elle existe déjà (sinon doublons à chaque run)
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR
    )
    print("Base vectorielle existante chargée.")
else:
    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR
    )
    print("Nouvelle base vectorielle créée.")
 
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)


# --- 3. Prompt template ---
prompt_template = """
Tu es un assistant chargé de répondre aux questions en te basant UNIQUEMENT sur le contexte fourni.

Le contexte est un document de règlement intérieur.

RÈGLES STRICTES :

* Utilise UNIQUEMENT les informations présentes dans le contexte.
* N’utilise AUCUNE connaissance externe.
* Si la réponse n’est pas clairement présente dans le contexte, réponds EXACTEMENT : JE NE SAIS PAS
* Sois précis, clair et concis.
* Si possible, cite la règle correspondante (Article, Chapitre, etc.)

COMPRÉHENSION :

* Fais preuve de tolérance face aux variations de langage (ex : “le” vs “la”, singulier/pluriel, petites fautes).
* Si la question est légèrement différente mais que le sens global correspond au contexte, considère-la comme valide.
* Base-toi sur le sens global de la phrase et non uniquement sur une correspondance exacte mot à mot.

LANGUE :

* Réponds uniquement en français.

<context>
{context}
</context>

<question>
{question}
</question>

Réponse :

"""
 
 
def RAG(query, llm=llm, prompt_template=prompt_template):
    """Récupère le contexte pertinent puis génère une réponse fondée dessus."""
    context_docs = retriever.invoke(query)
    context_list = [d.page_content for d in context_docs]
    context_for_query = ". ".join(context_list)
    prompt = prompt_template.format(context=context_for_query, question=query)
    resp = llm.invoke(prompt)
    return resp.content


# --- 4. Évaluation (LLM-as-a-judge) ---
groundedness_rater_system_message = """
Vous êtes un évaluateur expert chargé d'analyser la qualité des réponses générées par une IA.
 
On vous fournira une entrée structurée contenant :
- ###Question : la question posée par l'utilisateur
- ###Context : le contexte utilisé pour générer la réponse
- ###Answer : la réponse générée par l'IA
 
OBJECTIF :
Évaluer si la réponse est STRICTEMENT fondée sur le contexte fourni.
 
CRITÈRE (Groundedness) :
La réponse doit être entièrement dérivée du contexte.
Aucune information ne doit être inventée ou ajoutée.
 
ÉCHELLE DE NOTATION :
1 → Pas du tout fondée sur le contexte (hallucinations majeures)
2 → Faiblement fondée (beaucoup d'ajouts ou erreurs)
3 → Moyennement fondée (quelques erreurs ou ajouts)
4 → Majoritairement fondée (légères imprécisions)
5 → Totalement fondée (aucune hallucination)
 
INSTRUCTIONS :
1. Analysez la réponse phrase par phrase
2. Comparez chaque information avec le contexte
3. Identifiez :
   - informations correctes
   - informations absentes du contexte
   - éventuelles hallucinations
4. Expliquez clairement votre raisonnement
 
FORMAT DE SORTIE (OBLIGATOIRE) :
 
Analyse:
- ...
 
Points corrects:
- ...
 
Erreurs / hallucinations:
- ...
 
Score: X/5
"""
 
user_message_template = """
###Question
{question}
###Context
{context}
###Answer
{answer}
"""
 
groundness_checker = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model="nvidia/nemotron-3-ultra-550b-a55b:free"
)
 
 
def evaluate(question, context, answer, model=groundness_checker):
    """Évalue si la réponse est bien fondée sur le contexte."""
    prompt = f"""
    {groundedness_rater_system_message}

    USER :
    ###Question
    {question}

    ###Context
    {context}

    ###Answer
    {answer}
    """
    juge_response = model.invoke(prompt)
    return juge_response.content
