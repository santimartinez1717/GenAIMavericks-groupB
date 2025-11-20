import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from conf import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_core.documents import Document



def prepare_and_split_pdf(pdf_path: str):
    """
    Extrae el texto del PDF y lo divide en trozos (chunks) optimizados.
    """
    print("--- ETAPA 1: Extracción de PDF y Chunking ---")
    try:
        reader = PdfReader(pdf_path)
        raw_text = ''.join(page.extract_text() or '' for page in reader.pages)
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado en: {pdf_path}")
        return []

    if not raw_text:
        print("ERROR: El archivo PDF está vacío o no se pudo extraer el texto.")
        return []

    # Dividir el texto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    # Crear documentos de LangChain con metadatos
    docs = [Document(page_content=t, metadata={"source": pdf_path}) for t in text_splitter.split_text(raw_text)]
    
    print(f"✅ Documento dividido en {len(docs)} trozos.")
    return docs
