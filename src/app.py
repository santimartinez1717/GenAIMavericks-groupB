import streamlit as st
import os
from pypdf import PdfReader
from io import BytesIO

# Importamos las funciones de nuestro proyecto
from conf import PDF_PATH 
from chunking import prepare_and_split_pdf
from vectorstore import create_or_load_vectorstore
from rag import run_justicia_clara_agent

# --- Configuración de la Interfaz ---
st.set_page_config(page_title="Justicia Clara - Simplificador Legal", layout="wide")

# ==============================================================================
# 1. Función para la Extracción del PDF subido por el usuario
# ==============================================================================

def extract_text_from_uploaded_pdf(uploaded_file):
    """
    Lee un archivo PDF subido por Streamlit (BytesIO) y extrae todo el texto.
    """
    if uploaded_file is not None:
        try:
            # Creamos un objeto BytesIO a partir del archivo subido
            pdf_file = BytesIO(uploaded_file.read())
            
            reader = PdfReader(pdf_file)
            raw_text = ''.join(page.extract_text() or '' for page in reader.pages)
            
            if not raw_text:
                st.error("No se pudo extraer texto del PDF. Asegúrate de que no es solo una imagen.")
                return None
            
            return raw_text
        except Exception as e:
            st.error(f"Error al procesar el PDF: {e}")
            return None
    return None

# ==============================================================================
# 2. PROCESO DE SETUP RAG (Creación de la base de datos de la Guía)
# ==============================================================================

# Usamos st.cache_resource para que este proceso solo se ejecute una vez
@st.cache_resource
def initialize_rag_system():
    """
    Inicializa el sistema RAG (Guía de directrices).
    """
    try:
        if not os.path.exists(PDF_PATH):
            st.error(f"¡ERROR CRÍTICO! No se encontró el archivo de la Guía en: {PDF_PATH}")
            return None
            
        print("--- Configurando Sistema RAG ---")
        
        # 1. Preparar la Guía (Extracción y Chunking)
        document_chunks = prepare_and_split_pdf(PDF_PATH)
        
        if not document_chunks:
             st.error("Fallo al crear los chunks de la guía de directrices.")
             return None
             
        # 2. Crear/Cargar la Base de Datos Vectorial (ChromaDB)
        vector_db = create_or_load_vectorstore(document_chunks)
        
        return vector_db

    except Exception as e:
        st.error(f"Fallo en la inicialización del sistema RAG. Verifica tu clave API y librerías.")
        st.code(f"Error: {e}")
        return None

# ==============================================================================
# 3. INTERFAZ DE STREAMLIT (Main App)
# ==============================================================================

def main():
    st.title("⚖️ Justicia Clara: Simplificación Legal con IA")
    st.markdown(
        """
        Sube un documento judicial (PDF) para que nuestro Agente lo simplifique, 
        manteniendo la precisión legal gracias a la **Guía de redacción judicial clara**.
        """
    )
    
    # ----------------------------------------------------
    # A. Inicializar el sistema RAG
    # ----------------------------------------------------
    with st.spinner("Inicializando el Agente Justicia Clara..."):
        vector_db = initialize_rag_system()

    if vector_db is None:
        st.stop() # Detiene la ejecución si el sistema RAG falló.
    
    st.success("Sistema RAG listo. Base de datos de la Guía cargada.")
    
    # ----------------------------------------------------
    # B. Entrada del Usuario (PDF a simplificar)
    # ----------------------------------------------------
    st.header("1. Sube tu Documento Judicial (PDF)")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo PDF para simplificar.", 
        type="pdf"
    )

    documento_texto = None
    if uploaded_file:
        with st.spinner(f"Extrayendo texto de '{uploaded_file.name}'..."):
            documento_texto = extract_text_from_uploaded_pdf(uploaded_file)
        
        if documento_texto:
            st.success(f"Texto extraído: {len(documento_texto)} caracteres.")
            
    # ----------------------------------------------------
    # C. Botón de Procesamiento
    # ----------------------------------------------------
    st.header("2. Simplificar Documento")
    
    if st.button("🚀 Iniciar Simplificación Legal", type="primary", disabled=documento_texto is None):
        if documento_texto:
            st.markdown("---")
            st.subheader("Procesando...")
            
            with st.spinner("El Agente Justicia Clara está simplificando el lenguaje..."):
                try:
                    # Llama al agente RAG con el texto extraído del PDF subido
                    resultado_simplificado = run_justicia_clara_agent(vector_db, documento_texto)
                    
                    st.header("3. Resultado: Versión Ciudadana")
                    # Muestra la respuesta formateada en Markdown por el LLM
                    st.markdown(resultado_simplificado) 

                except Exception as e:
                    st.error("Error durante la generación de la respuesta.")
                    st.code(f"Detalle del Error: {e}")
        else:
            st.warning("Por favor, sube un documento PDF primero.")

if __name__ == '__main__':
    # Verificar la clave API al inicio
    if not os.getenv("GEMINI_API_KEY"):
        st.error(
            "La variable de entorno 'GEMINI_API_KEY' no está configurada. "
            "Por favor, configúrala en tu terminal: 'export GEMINI_API_KEY=\"TU_CLAVE\"'"
        )
    else:
        main()