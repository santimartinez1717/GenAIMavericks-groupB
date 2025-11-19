import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from conf import (PDF_PATH, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, GENERATION_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL)
from chunking import prepare_and_split_pdf
from vectorstore import create_or_load_vectorstore
from rag import run_justicia_clara_agent


doc_prueba = """"
JUZGADO DE PRIMERA INSTANCIA Nº 104 BIS DE MADRID
Calle Génova 10 , Planta 2ª - 28004
Tfno: 918352979 Fax: NO APLICA
juzpriminstancia104bis@madrid.org
42020310
NIG: 28.079.00.2-2024/0074722
Procedimiento: Procedimiento Ordinario 2101/2024
Materia: Resto de acciones individuales sobre condiciones generales de la contratación
Demandante: D./Dña. Maximo PROCURADOR D./Dña. ROBERTO DE HOYOS MENCIA Demandado: WIZINK
BANK, S.A.
PROCURADOR D./Dña. GEMMA DONDERIS DE SALAZAR
SENTENCIA Nº 432/2025
En Madrid, a 7 de febrero de 2025, Alicia Visitación Martín, Magistrada Juez del Juzgado de Primera Instancia
nº 104bis de Madrid, ha visto los presentes autos que se siguen en este Juzgado bajo el nº de procedimiento
2101/24, a instancias de Maximo , representado/a por procurador/a Roberto de Hoyos Mencía y asistido/a de
letrado/a Águeda María Martín Fernández frente a WIZINK BANK SA, representado por procurador/a Gemma
Donderis de Salazar y asistido de letrado/a David Castillejo Río.
ANTECEDENTES DE HECHO
PRIMERO-
. Por turno de reparto correspondió a este Juzgado demanda de juicio ordinario presentada por
el/a citado/a procurador/a en la representación referida, en la que tras exponer los hechos y fundamentos
de derecho que creyó aplicables terminaba en el suplico solicitando se dictara sentencia, conforme a los
pedimentos que se exponían y que se dan aquí por reproducidos.
SEGUNDO-
. Admitida a trámite la demanda se dio traslado a la entidad demandada, que contestó en tiempo
y forma. Citadas las partes, se celebró Audiencia Previa el día señalado al efecto, proponiéndose como única
prueba la documental, quedando los autos conclusos para sentencia.
TERCERO-
. En la tramitación del presente procedimiento se han observado las prescripciones legales.
FUNDAMENTOS DE DERECHO
PRIMERO-
.Pretensiones de las partes.
1
JURISPRUDENCIA
Por la parte actora se ejercita acción de declaración de nulidad del sistema de amortización revolving,
subsidiariamente, de la condición que recoge el interés remuneratorio y de manera subsidiaria, de la nulidad
del contrato por usura.
La parte demandada se opone."""

if __name__ == "__main__":
    
    # 1. Preparar el PDF
    document_chunks = prepare_and_split_pdf(PDF_PATH)
    
    if document_chunks:
        # 2. Crear/Cargar la Base de Datos Vectorial
        vector_db = create_or_load_vectorstore(document_chunks)
        
        # 3. Ejecutar el Agente de Simplificación
        run_justicia_clara_agent(vector_db, doc_prueba)
    else:
        print("\nEl proceso RAG se detuvo debido a un error en la extracción del PDF.")