# Asistente experto en machine learning y redes neuronales

Asistente de estudio basado en RAG (recuperación aumentada por generación). Responde preguntas sobre machine learning clásico y fundamentos de redes neuronales a partir de una colección de documentos propios, con memoria de conversación.

> Estado: en desarrollo. Completada la base de conocimiento (extracción, chunking, embeddings y ChromaDB). Pendientes: generación con Gemini, agente en LangGraph con memoria, interfaz en Streamlit y la justificación del system prompt.

## Dominio y documentos

Cinco PDF en español, guardados en `docs/`:

| Archivo | Contenido |
|---|---|
| `01_regresion_logistica.pdf` | Fundamentos, interpretación y evaluación de la regresión logística |
| `02_clasificacion.pdf` | SVM, ensembles y random forest, con ejemplos en Python |
| `03_algoritmos_machine_learning.pdf` | Panorama general de algoritmos (Cátedra iDanae) |
| `04_redes neuronales.pdf` | Fundamentos de redes neuronales y perceptrones multicapa (UPC) |
| `05_entrenamiento_redes.pdf` | Entrenamiento, pérdidas, optimización y regularización (Universidad de Oviedo) |

El asistente no entrena modelos: contesta con el contenido de estos documentos y debe indicar de dónde procede cada dato.

## Tecnologías

- Gemini: embeddings (`gemini-embedding-001`, 3.072 dimensiones) y generación.
- ChromaDB: base vectorial local persistente.
- LangGraph, con LangChain como base: agente y memoria.
- Jupyter Notebook y, como ampliación, Streamlit.

## Requisitos

- Python 3.12 o superior (desarrollado con 3.14.3).
- Una clave de la API de Gemini (Google AI Studio).
- Dependencias de `requirements.txt`.

## Instalación

Desde la carpeta del proyecto, en una terminal Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz con esta línea, sustituyendo el valor por tu clave (el archivo está excluido de Git y no debe compartirse):

```
GEMINI_API_KEY=tu_clave
```

## Ejecución

1. Abre `agente_experto_ml.ipynb` en VS Code o Jupyter y selecciona el kernel `.venv`.
2. Ejecuta las celdas en orden. Si existen los embeddings guardados en `datos_embeddings/`, no se vuelven a generar; las consultas sí llaman a la API de Gemini y consumen cuota.

## Estructura

- `docs/`: PDF originales.
- `datos_embeddings/`: embeddings y progreso en JSON. Cada archivo lleva en su nombre una huella del contenido y la configuración.
- `base_chroma/`: base vectorial local (se puede reconstruir desde `datos_embeddings/`).
- `agente_experto_ml.ipynb`: notebook del proyecto.

## System prompt

Pendiente de redactar y justificar.
