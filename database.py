import chromadb
from typing import List


client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

DOC_DB = {}

def save_raw_document(document_id: str, title: str, content: str):
    """Guarda el documento crudo antes de procesar embeddings."""
    DOC_DB[document_id] = {
        "title": title,
        "content": content
    }


def get_temp_document(document_id: str):
    """Recupera el documento crudo de la memoria temporal."""
    return DOC_DB.get(document_id)


def save_to_vector_db(db_id: str, original_doc_id: str, title: str, content_snippet: str, embedding: list):
    """
    Guarda un chunk procesado en ChromaDB.
    
    Parámetros:
    - db_id: ID único del chunk (ej: "doc123_0").
    - original_doc_id: ID del documento padre (ej: "doc123").
    - title: Título del documento original.
    - content_snippet: El texto del fragmento/chunk.
    - embedding: El vector generado para este fragmento.
    """
    collection.add(
        ids=[db_id],  # El ID único del chunk
        embeddings=[embedding], # El vector numérico
        documents=[content_snippet], # El texto visible para la búsqueda
        metadatas=[{
            "title": title,
            "document_id": original_doc_id # Guardamos el ID padre en metadata para agrupar si hace falta
        }]
    )


def search_chunks(query_embedding: list, n_results=3):
    """
    Busca los chunks más similares al embedding de la query.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        # Incluimos documentos y metadatos para poder construir la respuesta
        include=["documents", "metadatas", "distances"] 
    )
    return results