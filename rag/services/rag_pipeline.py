import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Load env
load_dotenv(override=True)
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY manquante.")

PDF_FILE = "./pdfs/reglements.pdf"
PERSIST_DIR = "./db_vector"
COLLECTION_NAME = "reglements_v1"

# --- LLM principal ---
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model="openai/gpt-oss-20b:free"
)

# --- LLM judge ---
groundness_checker = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model="nvidia/nemotron-3-ultra-550b-a55b:free"
)

#LLM Traducteur
traducteur = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
)

# --- Embeddings ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Charger ou créer la base ---
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR
    )
else:
    loader = PyPDFLoader(PDF_FILE)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = loader.load_and_split(text_splitter)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR
    )


# Retriever
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)



# --- Prompt ---
prompt_template = """
Tu es un assistant chargé de répondre aux questions en te basant UNIQUEMENT sur le contexte fourni.

Si la réponse n’est pas clairement présente, réponds EXACTEMENT : JE NE SAIS PAS

<context>
{context}
</context>

<question>
{question}
</question>

Réponse :
"""

def run_rag(question: str):
    docs = retriever.invoke(question)
    context = ". ".join([d.page_content for d in docs])

    prompt = prompt_template.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)
    return response.content, context


def traducteur_francais(text):
    prompt = f"""
        Tu es un traducteur expert bilingue (wolof ↔ français), spécialisé dans les textes officiels et réglementaires.

        MISSION :
        Traduire le texte wolof fourni en français de manière STRICTE et FIDÈLE.

        RÈGLES OBLIGATOIRES :

        1. Ne modifie PAS le sens du texte
        2. Ne simplifie PAS le contenu
        3. Ne résume PAS
        4. Ne rajoute AUCUNE information
        5. Respecte le ton formel du texte
        6. Traduis chaque phrase avec précision
        7. Si un terme n’a pas d’équivalent exact en français, garde-le en wolof

        FORMAT DE SORTIE :

        - Donne UNIQUEMENT la traduction finale
        - AUCUNE explication
        - AUCUN commentaire
        - AUCUN texte supplémentaire

        IMPORTANT :
        Prends le temps de bien comprendre le texte avant de traduire.

        TEXTE :
        {text}
    """
    return traducteur.invoke(prompt).content.strip()


def traducteur_wolof(text):
    prompt = f"""
        Tu es un traducteur expert bilingue (français ↔ wolof), spécialisé dans les textes officiels et réglementaires.

        MISSION :
        Traduire le texte français fourni en wolof de manière STRICTE et FIDÈLE.

        RÈGLES OBLIGATOIRES :

        1. Ne modifie PAS le sens du texte
        2. Ne simplifie PAS le contenu
        3. Ne résume PAS
        4. Ne rajoute AUCUNE information
        5. Respecte le ton formel du texte
        6. Traduis chaque phrase avec précision
        7. Si un terme n’a pas d’équivalent exact en wolof, garde-le en français

        FORMAT DE SORTIE :

        - Donne UNIQUEMENT la traduction finale
        - AUCUNE explication
        - AUCUN commentaire
        - AUCUN texte supplémentaire

        IMPORTANT :
        Prends le temps de bien comprendre le texte avant de traduire.

        TEXTE :
        {text}
    """
    return traducteur.invoke(prompt).content.strip()


def multilingual_rag(user_question):
    # 1. Traduction vers français
    question_fr = traducteur_francais(user_question)

    # 2. RAG
    answer_fr, context = run_rag(question_fr)

    # 3. Traduction vers wolof
    answer_wolof = traducteur_wolof(answer_fr)

    return {
        "question_originale": user_question,
        "question_fr": question_fr,
        "reponse_fr": answer_fr,
        "reponse_wolof": answer_wolof,
        "context": context
    }

# --- Evaluation ---
def evaluate(question: str, context: str, answer: str):

    system_message = """Évalue si la réponse est fondée sur le contexte. Score de 1 à 5 + explication."""

    prompt = f"""
        {system_message}

        Question:
        {question}

        Context:
        {context}

        Answer:
        {answer}
    """

    response = groundness_checker.invoke(prompt)
    return response.content

