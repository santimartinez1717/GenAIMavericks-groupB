"""
9 Reglas de Simplificación
"""

import re

class SimplificationRules:
    """Implementación de las 9 reglas oficiales"""
    
    def apply_all_rules(self, text: str) -> str:
        """Aplicar todas las reglas"""
        result = text
        result = self.rule_1_lists(result)
        result = self.rule_2_capitals(result)
        result = self.rule_3_dates(result)
        result = self.rule_4_legal_refs(result)
        result = self.rule_5_greetings(result)
        result = self.rule_6_terminology(result)
        result = self.rule_7_short_sentences(result)
        result = self.rule_8_structure(result)
        result = self.rule_9_verbs(result)
        return result
    
    def rule_1_lists(self, text: str) -> str:
        """Listas verticales"""
        text = re.sub(r'(\d+\).*?),\s*(\d+\))', r'\1\n\2', text)
        return text
    
    def rule_2_capitals(self, text: str) -> str:
        """Mayúsculas innecesarias"""
        words = ['VISTO', 'CONSIDERANDO', 'FALLO', 'ANTECEDENTES']
        for word in words:
            pattern = r'\b' + word + r'\b'
            replacement = word.capitalize()
            text = re.sub(pattern, replacement, text)
        return text
    
    def rule_3_dates(self, text: str) -> str:
        """Fechas legibles"""
        months = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
        
        def replace_date(match):
            day, month, year = match.groups()
            month_idx = int(month) - 1
            if 0 <= month_idx < 12:
                return f"{int(day)} de {months[month_idx]} de {year}"
            return match.group(0)
        
        text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', replace_date, text)
        return text
    
    def rule_4_legal_refs(self, text: str) -> str:
        """Referencias legales"""
        # Placeholder
        return text
    
    def rule_5_greetings(self, text: str) -> str:
        """Saludos modernos"""
        replacements = {
            r'Excelentísimo\s+Señor': 'Señor',
            r'Ilustrísimo\s+Señor': 'Señor',
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def rule_6_terminology(self, text: str) -> str:
        """Terminología clara"""
        terms = {
            r'de conformidad con': 'según',
            r'a tenor de': 'según',
            r'en virtud de': 'por',
        }
        
        for pattern, replacement in terms.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def rule_7_short_sentences(self, text: str) -> str:
        """Oraciones cortas"""
        # Placeholder
        return text
    
    def rule_8_structure(self, text: str) -> str:
        """Estructura S+V+O"""
        # Placeholder
        return text
    
    def rule_9_verbs(self, text: str) -> str:
        """Verbos modernos"""
        verbs = {
            r'\bhubiere\b': 'haya',
            r'\bfuere\b': 'sea',
        }
        
        for pattern, replacement in verbs.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
