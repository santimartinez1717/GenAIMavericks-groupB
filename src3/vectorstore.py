import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from conf import CHROMA_DB_DIR, EMBEDDING_MODEL

def create_or_load_vectorstore(docs: list[Document]):
    """
    Crea la base de datos vectorial con ChromaDB y los embeddings de Google.
    Si el directorio ya existe, carga la base de datos existente.
    """
    print("\n--- ETAPA 2: Creación/Carga de Vectorstore ---")

    # Inicializar el modelo de Embeddings
    embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_DB_DIR):
        print(f"Cargando base de datos existente desde: {CHROMA_DB_DIR}")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR, 
            embedding_function=embeddings_model
        )
    else:
        print("Creando nueva base de datos vectorial...")
        # Almacenar en ChromaDB
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings_model, 
            persist_directory=CHROMA_DB_DIR
        )
        vectorstore.persist()
        print(f"✅ Base de datos vectorial creada y guardada en {CHROMA_DB_DIR}")
        
    return vectorstore