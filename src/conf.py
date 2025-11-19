import os
# --- Asegúrate de que la clave API de Gemini esté configurada ---
# La forma más segura es configurar la variable de entorno:
# export GEMINI_API_KEY="TU_CLAVE_AQUI"
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError(
        "La variable de entorno 'GEMINI_API_KEY' no está configurada. "
        "Por favor, configúrala antes de ejecutar."
    )

# --- Rutas y Nombres de Archivos ---
PDF_PATH = "Guia_de_redaccion_judicial_clara.pdf"
CHROMA_DB_DIR = "./chroma_db"

# --- Parámetros de RAG (Chunking) ---
# Tamaño óptimo de los "trozos" para capturar el contexto de una regla
CHUNK_SIZE = 700 
CHUNK_OVERLAP = 70 

# --- Modelo LLM y Embeddings ---
EMBEDDING_MODEL = "text-embedding-004" # Recomendado para RAG
GENERATION_MODEL = "gemini-2.5-pro"   # Rápido y potente para la tarea
LLM_TEMPERATURE = 0.1                   # Baja para menos 'creatividad' y más fidelidad
