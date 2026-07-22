import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Load env
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

PDF_FILE = "./pdfs/reglements.pdf"
PERSIST_DIR = "./db_test"
COLLECTION_NAME = "reglements_v1"

# LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model="openai/gpt-oss-20b:free"
)

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Charger ou créer la base UNE SEULE FOIS
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

# Prompt
PROMPT_TEMPLATE = """
Tu es un assistant.

Réponds uniquement avec le contexte fourni.
Si tu ne sais pas, dis "I DO NOT KNOW".

<context>
{context}
</context>

<question>
{question}
</question>
"""

def run_rag(query: str):
    docs = retriever.invoke(query)
    context = ". ".join([d.page_content for d in docs])

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=query
    )

    response = llm.invoke(prompt)
    return response.content

