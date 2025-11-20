import streamlit as st
import sys
from pathlib import Path
import time

# Añadir src al path
sys.path.append(str(Path(__file__).parent))

from src.dual_rag_system import DualRAGSystem
from src.utils import extract_text_from_pdf, save_output

# Configuración de página
st.set_page_config(
    page_title="Justicia Clara",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #21808d 0%, #1d7480 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    .badge-guia {
        background: rgba(33, 128, 141, 0.15);
        color: #21808d;
        border: 1px solid #21808d;
    }
    .badge-cendoj {
        background: rgba(230, 129, 97, 0.15);
        color: #e68161;
        border: 1px solid #e68161;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ Justicia Clara</h1>
    <p>Sistema de Simplificación de Documentos Judiciales con IA</p>
    <p style="font-size: 0.9rem; opacity: 0.9;">RAG Dual: Guía Oficial + CENDOJ</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    use_cendoj = st.checkbox(
        "Usar contexto CENDOJ",
        value=True,
        help="Incluir sentencias reales de CENDOJ en el RAG"
    )
    
    top_k = st.slider(
        "Ejemplos a recuperar",
        min_value=3,
        max_value=10,
        value=5,
        help="Número de ejemplos relevantes a usar"
    )
    
    st.markdown("---")
    st.header("📋 Reglas Aplicadas")
    
    reglas = [
        "1. Listas y Enumeraciones",
        "2. Mayúsculas Innecesarias",
        "3. Fechas y Cifras",
        "4. Referencias Legales",
        "5. Salutaciones Modernas",
        "6. Terminología Clara",
        "7. Oraciones Cortas",
        "8. Estructura Clásica",
        "9. Verbos Modernos"
    ]
    
    for regla in reglas:
        st.markdown(f"✓ {regla}")

# Inicializar sistema (cachear)
@st.cache_resource
def init_system(use_cendoj_flag):
    guia_path = Path("data/Guia_de_redaccion_judicial_clara.pdf")
    
    if not guia_path.exists():
        st.error(f"❌ No se encuentra la Guía en: {guia_path}")
        return None
    
    with st.spinner("🚀 Inicializando Sistema RAG Dual..."):
        return DualRAGSystem(
            guia_path=str(guia_path),
            use_cendoj=use_cendoj_flag
        )

rag_system = init_system(use_cendoj)

if rag_system is None:
    st.stop()

# Main area
tab1, tab2, tab3 = st.tabs(["📄 Cargar Documento", "📊 Resultados", "ℹ️ Información"])

with tab1:
    st.header("📄 Cargar Documento Judicial")
    
    uploaded_file = st.file_uploader(
        "Subir sentencia o auto (PDF o TXT)",
        type=['pdf', 'txt'],
        help="Máximo 10MB"
    )
    
    if uploaded_file:
        # Mostrar info del archivo
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Archivo:** {uploaded_file.name}")
        with col2:
            st.info(f"**Tamaño:** {uploaded_file.size / 1024:.2f} KB")
        
        # Extraer texto
        if uploaded_file.name.endswith('.pdf'):
            texto_original = extract_text_from_pdf(uploaded_file)
        else:
            texto_original = uploaded_file.read().decode('utf-8')
        
        # Mostrar preview
        with st.expander("👁️ Vista previa del texto original"):
            st.text_area("", texto_original[:1000] + "...", height=200, disabled=True)
        
        # Botón de simplificación
        if st.button("🔄 Simplificar Documento", type="primary", use_container_width=True):
            with st.spinner("⏳ Procesando con IA..."):
                start_time = time.time()
                
                # Simplificar
                resultado = rag_system.simplificar(texto_original)
                
                end_time = time.time()
                
                # Guardar en session state
                st.session_state['resultado'] = resultado
                st.session_state['tiempo_procesamiento'] = end_time - start_time
                st.session_state['texto_original'] = texto_original
                
                st.success(f"✅ Documento simplificado en {end_time - start_time:.2f}s")
                st.info("👉 Ve a la pestaña **Resultados** para ver el documento simplificado")

with tab2:
    if 'resultado' in st.session_state:
        resultado = st.session_state['resultado']
        texto_original = st.session_state['texto_original']
        
        st.header("📊 Resultados de la Simplificación")
        
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            palabras_orig = len(texto_original.split())
            st.metric("Palabras Originales", palabras_orig)
        
        with col2:
            palabras_simp = len(resultado['simplificado'].split())
            st.metric("Palabras Simplificadas", palabras_simp)
        
        with col3:
            st.metric(
                "Reducción",
                f"{(1 - palabras_simp/palabras_orig) * 100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Tiempo",
                f"{st.session_state['tiempo_procesamiento']:.2f}s"
            )
        
        # Fuentes usadas
        st.markdown("---")
        st.subheader("📚 Fuentes Utilizadas")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<span class="source-badge badge-guia">Guía: {resultado["fuentes"]["ejemplos_guia"]} ejemplos</span>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<span class="source-badge badge-cendoj">CENDOJ: {resultado["fuentes"]["contextos_cendoj"]} contextos</span>',
                unsafe_allow_html=True
            )
        
        # Ejemplos recuperados
        with st.expander("🔍 Ver Ejemplos de la Guía Utilizados"):
            for i, ej in enumerate(resultado['resultados_rag']['guia'], 1):
                st.markdown(f"**Ejemplo {i}** (relevancia: {ej['similarity']:.1%})")
                st.markdown(f"❌ **Original:** {ej['documento'][:200]}...")
                st.markdown(f"✅ **Simplificado:** {ej['metadata'].get('simplificado', 'N/A')[:200]}...")
                st.markdown("---")
        
        if use_cendoj:
            with st.expander("⚖️ Ver Contextos CENDOJ Utilizados"):
                for i, ctx in enumerate(resultado['resultados_rag']['cendoj'], 1):
                    st.markdown(f"**Contexto {i}** (relevancia: {ctx['similarity']:.1%})")
                    st.markdown(f"{ctx['documento'][:300]}...")
                    st.caption(f"Fuente: {ctx['metadata'].get('organo', 'N/A')}")
                    st.markdown("---")
        
        # Comparación lado a lado
        st.markdown("---")
        st.subheader("📝 Comparación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📜 Texto Original")
            st.text_area("", texto_original, height=500, disabled=True, key="orig")
        
        with col2:
            st.markdown("### ✨ Texto Simplificado")
            st.text_area("", resultado['simplificado'], height=500, disabled=True, key="simp")
        
        # Botón de descarga
        st.download_button(
            label="💾 Descargar Versión Simplificada",
            data=resultado['simplificado'],
            file_name="documento_simplificado.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    else:
        st.info("👈 Sube un documento en la pestaña **Cargar Documento** primero")

with tab3:
    st.header("ℹ️ Información del Sistema")
    
    st.markdown("""
    ### 🎯 ¿Qué es Justicia Clara?
    
    Sistema de simplificación automática de documentos judiciales que utiliza:
    
    - **RAG Dual**: Combina la Guía de Redacción Judicial Clara del Ministerio de Justicia 
      con sentencias reales de CENDOJ
    - **IA Local**: Procesamiento con Ollama (Llama 2 / Mistral)
    - **9 Reglas Oficiales**: Implementación completa de las recomendaciones ministeriales
    
    ### 🔬 Arquitectura RAG Dual
    
    ```
    Usuario → Documento
        ↓
    Búsqueda Paralela
        ↙         ↘
    RAG Guía    RAG CENDOJ
        ↓             ↓
    Ejemplos    Contexto Real
        ↘         ↙
      Prompt Enriquecido
            ↓
        LLM (Llama 2)
            ↓
    Texto Simplificado
    ```
    
    ### 📊 Beneficios
    
    - ✅ Accesibilidad ciudadana
    - ✅ Reducción de reclamaciones
    - ✅ Transparencia judicial
    - ✅ Cumplimiento Justicia 2030
    
    ### 🤝 Créditos
    
    - Ministerio de Justicia - Guía de Redacción Judicial Clara
    - CGPJ - CENDOJ
    - Proyecto académico - Universidad
    """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #626c71;">⚖️ Justicia Clara v1.0 - Sistema RAG Dual</div>',
    unsafe_allow_html=True
)
