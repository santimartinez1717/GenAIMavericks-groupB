"""
Utilidades
"""

import PyPDF2
from pathlib import Path

def extract_text_from_pdf(file) -> str:
    """Extraer texto de PDF"""
    try:
        pdf = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise Exception(f"Error extrayendo PDF: {e}")

def save_output(text: str, output_path: Path):
    """Guardar texto simplificado"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
