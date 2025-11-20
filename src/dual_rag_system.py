"""
Sistema RAG Dual para Justicia Clara
Combina Guía Oficial + CENDOJ
"""

from sentence_transformers import SentenceTransformer
import chromadb
from typing import Dict, List, Optional
import PyPDF2
import re
from pathlib import Path
# Al inicializar (solo una vez):
REGLAS_SIMPLIFICADAS = Path("data/guia/reglas_simplificadas.txt").read_text(encoding="utf-8")


class DualRAGSystem:
    """
    Sistema RAG Dual:
    - RAG 1: Guía Oficial (ejemplos de simplificación)
    - RAG 2: CENDOJ (sentencias reales)
    """
    
    def __init__(self, guia_path: str, use_cendoj: bool = True):
        print("🚀 Inicializando Sistema RAG Dual...")
        
        # Modelo de embeddings
        self.encoder = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        # ChromaDB client
        self.client = chromadb.Client()
        
        # RAG 1: Guía
        self.rag_guia = self.client.get_or_create_collection(
            name="rag_guia_simplificacion",
            metadata={"hnsw:space": "cosine"}
        )
        
        # RAG 2: CENDOJ
        self.rag_cendoj = self.client.get_or_create_collection(
            name="rag_cendoj_sentencias",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Cargar datos
        self._index_guia(guia_path)
        
        if use_cendoj:
            self._index_cendoj_mock()  # Mock por ahora
        
        print("✅ Sistema RAG Dual listo")
    
    def _index_guia(self, guia_path: str):
        """Indexar ejemplos de la Guía"""
        print("📚 Indexando Guía Oficial...")
        
        ejemplos = self._extract_ejemplos_guia(guia_path)
        
        for i, ejemplo in enumerate(ejemplos):
            embedding = self.encoder.encode(ejemplo['original'])
            
            self.rag_guia.add(
                embeddings=[embedding.tolist()],
                documents=[ejemplo['original']],
                metadatas=[{
                    'simplificado': ejemplo['simplificado'],
                    'regla': ejemplo['regla']
                }],
                ids=[f"guia_{i}"]
            )
        
        print(f"✅ {len(ejemplos)} ejemplos indexados")
    
    def _extract_ejemplos_guia(self, pdf_path: str) -> List[Dict]:
        """Extraer ejemplos de la Guía"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                texto = ""
                for page in pdf.pages:
                    texto += page.extract_text()
            
            # Pattern para ejemplos
            pattern = r'Versión no recomendada:?\s*(.*?)\s*Versión alternativa:?\s*(.*?)(?=Versión|$)'
            matches = re.findall(pattern, texto, re.DOTALL | re.IGNORECASE)
            
            ejemplos = []
            for i, (orig, simp) in enumerate(matches):
                ejemplos.append({
                    'id': f'guia_{i}',
                    'original': orig.strip()[:500],
                    'simplificado': simp.strip()[:500],
                    'regla': self._inferir_regla(orig, simp)
                })
            
            return ejemplos
        
        except Exception as e:
            print(f"⚠️ Error extrayendo Guía: {e}")
            return self._get_mock_ejemplos()
    
    def _get_mock_ejemplos(self) -> List[Dict]:
        """Ejemplos mock si falla la extracción"""
        return [
            {
                'id': 'mock_1',
                'original': 'VISTO el contenido de las actuaciones',
                'simplificado': 'Visto el contenido de las actuaciones',
                'regla': 'mayusculas'
            },
            {
                'id': 'mock_2',
                'original': 'de conformidad con lo establecido en el artículo 25.3',
                'simplificado': 'según el artículo 25.3',
                'regla': 'terminologia'
            },
            {
                'id': 'mock_3',
                'original': 'el demandante procede a solicitar',
                'simplificado': 'el demandante solicita',
                'regla': 'oraciones_cortas'
            }
        ]
    
    def _inferir_regla(self, orig: str, simp: str) -> str:
        """Inferir regla aplicada"""
        if orig.isupper() and not simp.isupper():
            return 'mayusculas'
        elif 'conformidad' in orig.lower():
            return 'terminologia'
        elif len(orig.split()) > len(simp.split()) * 1.3:
            return 'oraciones_cortas'
        return 'general'
    
    def _index_cendoj_mock(self):
        """Indexar mock de CENDOJ"""
        print("⚖️ Indexando mock CENDOJ...")
        
        ejemplos_mock = [
            {
                'texto': 'El Juzgado de Primera Instancia número 18 de Madrid ha visto el procedimiento ordinario...',
                'metadata': {
                    'organo': 'Juzgado Primera Instancia Madrid',
                    'fecha': '2023-09-23',
                    'procedimiento': 'Ordinario'
                }
            },
            {
                'texto': 'Vistos los autos de juicio ordinario seguidos ante este Juzgado bajo el número...',
                'metadata': {
                    'organo': 'Juzgado Primera Instancia Madrid',
                    'fecha': '2023-02-27',
                    'procedimiento': 'Ordinario'
                }
            }
        ]
        
        for i, ej in enumerate(ejemplos_mock):
            embedding = self.encoder.encode(ej['texto'])
            
            self.rag_cendoj.add(
                embeddings=[embedding.tolist()],
                documents=[ej['texto']],
                metadatas=ej['metadata'],
                ids=[f"cendoj_{i}"]
            )
        
        print(f"✅ {len(ejemplos_mock)} contextos CENDOJ indexados")
    
    def retrieve_hybrid(self, query: str, top_k: int = 5) -> Dict:
        """Búsqueda híbrida en ambos RAGs"""
        query_embedding = self.encoder.encode(query)
        
        # Buscar en Guía
        res_guia = self.rag_guia.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, 3)
        )
        
        # Buscar en CENDOJ
        res_cendoj = self.rag_cendoj.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, 2)
        )
        
        return {
            'guia': self._format_results(res_guia),
            'cendoj': self._format_results(res_cendoj)
        }
    
    def _format_results(self, results) -> List[Dict]:
        """Formatear resultados"""
        formatted = []
        
        if not results['ids'] or len(results['ids'][0]) == 0:
            return formatted
        
        for i in range(len(results['ids'][0])):
            formatted.append({
                'documento': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i]
            })
        
        return formatted
        
    def build_prompt(self, user_text: str) -> tuple:
        results = self.retrieve_hybrid(user_text)
        
        prompt = f"""Eres experto en simplificar documentos judiciales.

    ════════════ Reglas resumidas oficiales ════════════

        {REGLAS_SIMPLIFICADAS}

    ════════════ Ejemplos de la Guía Oficial ════════════

    """
        for i, ej in enumerate(results['guia'], 1):
            prompt += f"""Ejemplo {i}:
    ❌ Original: {ej['documento']}
    ✅ Simplificado: {ej['metadata'].get('simplificado', 'N/A')}

    """

        prompt += f"""
    ════════════ Contextos de Sentencias CENDOJ ════════════

    """
        for i, ctx in enumerate(results['cendoj'], 1):
            prompt += f"""Contexto {i}: {ctx['documento'][:200]}...

    """

        prompt += f"""
    ════════════ TEXTO A SIMPLIFICAR ════════════

        {user_text}

    INSTRUCCIONES:
    1. Aplica las reglas oficiales mostradas arriba
    2. Sigue los patrones de los ejemplos
    3. Mantén el significado jurídico exacto
    4. Usa lenguaje claro y accesible

    VERSIÓN SIMPLIFICADA:
    """

        return prompt, results

    def simplificar(self, texto: str) -> Dict:
        """Simplificar documento"""
        from src.llm_handler import LLMHandler
        
        # Construir prompt
        prompt, results = self.build_prompt(texto)
        
        # Generar con LLM
        llm = LLMHandler()
        simplificado = llm.generate(prompt)
        
        return {
            'original': texto,
            'simplificado': simplificado,
            'fuentes': {
                'ejemplos_guia': len(results['guia']),
                'contextos_cendoj': len(results['cendoj'])
            },
            'resultados_rag': results
        }
