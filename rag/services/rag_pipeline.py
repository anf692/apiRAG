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
Tu es un assistant qui répond aux questions en se basant UNIQUEMENT sur le contexte fourni.

Si la réponse n’est pas clairement présente dans le contexte, réponds EXACTEMENT :
JE NE SAIS PAS

REGLES STRICTES:
- Utilise UNIQUEMENT les informations du contexte
- Ne rajoute aucune connaissance externe
- Ne fais aucune supposition
- Ne traduis pas mot à mot, reformule naturellement

LANGUE:
- Si la question est en français → réponds uniquement en français
- Si la question est en wolof → réponds uniquement en wolof

Wolof (TRÈS IMPORTANT):
- Utilise un wolof SIMPLE, NATUREL et CORRECT
- N’invente pas de mots
- N’utilise pas de traduction littérale du français
- Utilise des phrases compréhensibles comme dans la vie réelle au Sénégal

FORMAT:
- Donne une réponse claire et directe
- Pas de mélange de langues

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

