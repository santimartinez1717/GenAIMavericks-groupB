from setuptools import setup, find_packages

setup(
    name="justicia-clara",
    version="1.0.0",
    author="Santiago Martínez",
    author_email="sant.martinez2004@gmail.com",
    description="Sistema de Simplificación de Documentos Judiciales con IA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/justicia-clara",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=open("requirements.txt").read().splitlines(),
)
