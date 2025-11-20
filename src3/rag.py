import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

from conf import (
    PDF_PATH, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    GENERATION_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL
)

# -----------------------------------------------------------------------------------
# NUEVO RAG (LCEL) — Sustituye a RetrievalQA (que ya no existe en LangChain 1.0)
# -----------------------------------------------------------------------------------


def build_rag_chain(retriever):
    """
    Construye el pipeline RAG moderno usando LCEL.
    """

    prompt = PromptTemplate(
        template="""
Eres el "Agente Justicia Clara", un experto en lenguaje legal. Tu objetivo es:
1. Simplificar el texto judicial proporcionado en el 'DOCUMENTO A SIMPLIFICAR'.
2. DEBES utilizar EXCLUSIVAMENTE el contexto y las reglas de redacción de la
   'Guía de redacción judicial clara' que se te proporciona en el contexto recuperado.
3. Mantén fidelidad absoluta al significado legal original.

FORMATO DE SALIDA:

## ⚖️ Documento Judicial Simplificado

### 📝 Resumen Ejecutivo para el Ciudadano
(Máximo 3 puntos clave: Quién, Qué pasó y Qué se decidió).

### 🗣️ Texto Simplificado (Párrafo a Párrafo)

### 💡 Glosario Rápido
(5 términos complejos explicados para un ciudadano)

------------------------------------------
Contexto Recuperado:
{context}

DOCUMENTO A SIMPLIFICAR:
{question}

Respuesta:
""",
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    # Une fragmentos (documents → texto)
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Nuevo pipeline LCEL
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain


# -----------------------------------------------------------------------------------
# AGENTE PRINCIPAL
# -----------------------------------------------------------------------------------

def run_justicia_clara_agent(vectorstore, texto_a_simplificar: str):
    """
    Define el pipeline RAG y ejecuta la simplificación para un texto dado.
    """
    print("\n--- ETAPA 3: Ejecución del Agente Justicia Clara (RAG) ---")

    # Retriever y construcción de la cadena RAG (igual que antes)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_chain = build_rag_chain(retriever)

    print("\n--- SIMPLIFICANDO TEXTO DE ENTRADA ---")

    # En lugar de usar una variable interna, usamos el parámetro de entrada
    result = rag_chain.invoke(texto_a_simplificar) 

    # Respuesta final
    print("\n\n===== RESPUESTA GENERADA =====\n")
    print(result) # Asumiendo que el resultado es un objeto LangChain con 'output' o 'result'

    print("\n---------------------------------------------------------")
    print("ℹ️  No se muestran source_documents porque LCEL ya no devuelve metadata automáticamente.")
    print("Si quieres incluirlos, te puedo añadir la extracción manual de vectorstore.")

    return result.content

