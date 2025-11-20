"""
Manejador de LLM (Ollama)
"""

class LLMHandler:
    """Interfaz con Ollama"""
    
    def __init__(self, model: str = "llama2"):
        self.model = model
        self._check_ollama()
    
    def _check_ollama(self):
        """Verificar que Ollama esté disponible"""
        try:
            import ollama
            self.ollama = ollama
            print(f"✅ Ollama conectado (modelo: {self.model})")
        except ImportError:
            print("⚠️ Ollama no disponible, usando modo mock")
            self.ollama = None
    
    def generate(self, prompt: str) -> str:
        """Generar con LLM"""
        if self.ollama:
            try:
                response = self.ollama.generate(
                    model=self.model,
                    prompt=prompt
                )
                return response['response']
            except Exception as e:
                print(f"⚠️ Error LLM: {e}, usando reglas básicas")
                return self._fallback_simplification(prompt)
        else:
            return self._fallback_simplification(prompt)
    
    def _fallback_simplification(self, prompt: str) -> str:
        """Simplificación básica sin LLM"""
        from src.simplification_rules import SimplificationRules
        
        # Extraer texto del prompt
        lines = prompt.split('\n')
        texto = ""
        capturando = False
        
        for line in lines:
            if 'TEXTO A SIMPLIFICAR' in line:
                capturando = True
                continue
            if capturando and 'INSTRUCCIONES' in line:
                break
            if capturando:
                texto += line + "\n"
        
        # Aplicar reglas
        rules = SimplificationRules()
        return rules.apply_all_rules(texto.strip())
