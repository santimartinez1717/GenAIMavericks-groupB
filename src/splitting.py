from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --- Configuración ---
# La ruta a tu archivo de 50 páginas
PDF_PATH = "Guia_redaccion_judicial_clara.pdf"
# Definición del tamaño óptimo de los "trozos" de texto
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50 # Superposición para mantener el contexto entre trozos

def preparar_guia_para_rag(pdf_path: str):
    """
    Extrae el texto del PDF y lo divide en trozos.
    """
    # 1. Extracción del texto del PDF
    print(f"Leyendo documento: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {pdf_path}")
        return []

    # 2. División del texto (Chunking)
    # RecursiveCharacterTextSplitter es ideal para documentos estructurados
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    # Crea los trozos de texto
    chunks = text_splitter.create_documents([raw_text])
    
    # Extraer el contenido de los trozos para usarlo como lista de strings
    chunk_contents = [chunk.page_content for chunk in chunks]
    
    print(f"Documento dividido en {len(chunk_contents)} trozos.")
    return chunk_contents

# Ejecución del proceso
guia_chunks = preparar_guia_para_rag(PDF_PATH)
# Puedes ver los primeros trozos para validar:
# print("\nPrimer Chunk de ejemplo:")
# print(guia_chunks[0])