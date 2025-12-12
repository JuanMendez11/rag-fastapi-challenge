from fastapi import FastAPI, HTTPException
from schemas import DocumentUploadRequest, DocumentUploadResponse, GenerateEmbeddingsRequest, GenerateEmbeddingsResponse, SearchRequest, SearchResultItem, SearchResponse, AskQuestionRequest, AskQuestionResponse
from database import save_raw_document, get_temp_document, save_to_vector_db, search_chunks
from services import split_text_into_chunks, get_embedding, generate_answer_with_context
import uuid
import logging
import sys
import os

app = FastAPI(title="Sistema RAG API", version="1.0.0")


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/traza_rag.log"), # Guarda en un archivo
        logging.StreamHandler(sys.stdout)     # Muestra en la terminal
    ]
)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logger = logging.getLogger("API_RAG")


@app.post("/upload", response_model=DocumentUploadResponse, status_code=201)
def upload(request: DocumentUploadRequest):
    """Carga un documento en el sistema."""
    logger.info(f"--> Solicitud de carga recibida para el documento: '{request.title}'")
    
    # Validamos que title no esté vacio
    if not request.title.strip():
        raise HTTPException(status_code=400, detail="El título no puede estar vacío.")
    # Validamos que content no esté vacio
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="El contenido no puede estar vacío.")

    # Generamos id
    document_id = str(uuid.uuid4())

    # Lo guardamos en una lista por ahora
    save_raw_document(document_id, request.title, request.content)

    logger.info(f"Documento cargado exitosamente. ID asignado: {document_id}")

    return {
        "message": "Document uploaded successfully",
        "document_id": document_id
    }


@app.post("/generate-embeddings", response_model=GenerateEmbeddingsResponse)
def generate_embeddings(request: GenerateEmbeddingsRequest):
    logger.info(f"--> Iniciando generación de embeddings para DOC ID: {request.document_id}")
    doc_id = request.document_id
    doc_data = get_temp_document(doc_id)
    
    if not doc_data:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    chunks = split_text_into_chunks(doc_data["content"], chunk_size=500, chunk_overlap=50)

    try:
        # Procesamos cada chunk
        for i, chunk_text in enumerate(chunks):
            # a. Generar vector con API externa
            vector = get_embedding(chunk_text)
            
            # b. Crear ID único para el chunk (ej: "doc123_0", "doc123_1")
            chunk_id = f"{doc_id}_{i}"
            
            # c. Guardar en ChromaDB
            # IMPORTANTE: Guardamos el 'chunk_text' como el documento visible
            # y el 'doc_id' original en la metadata para saber de dónde vino.
            save_to_vector_db(
                db_id=chunk_id,
                original_doc_id=doc_id, # Metadata clave
                title=doc_data["title"],
                content_snippet=chunk_text,
                embedding=vector
            )
        logger.info(f"Embeddings guardados correctamente en ChromaDB para DOC ID: {request.document_id}, se crearon {len(chunks)} chunks.")

    except Exception as e:
        # Manejo de error controlado según requisito
        logger.error(f"FALLO CRÍTICO en generación de embeddings: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="El servicio externo no pudo procesar la solicitud en este momento."
        )

    return {
        "message": f"Embeddings generated successfully for document {doc_id}"
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    logger.info(f"--> Búsqueda iniciada. Query: '{request.query}'")

    query_text = request.query
    
    # Generar embedding de la consulta
    try:
        query_vector = get_embedding(query_text)
    except Exception:
         raise HTTPException(
            status_code=500, 
            detail="El servicio externo no pudo procesar la solicitud."
        )

    # Buscar en ChromaDB
    search_results = search_chunks(query_vector, n_results=3) 

    # Formatear la respuesta
    ids = search_results['ids'][0]
    documents = search_results['documents'][0]
    metadatas = search_results['metadatas'][0]
    distances = search_results['distances'][0]

    formatted_results = []

    # Iteramos sobre los resultados encontrados
    for i in range(len(ids)):
        
        # Calcular Score: Convertir distancia a similitud (aproximado)
        score = 1 - distances[i] 

        item = SearchResultItem(
            # OJO: Aquí recuperamos el ID del documento PADRE que guardamos en metadata
            document_id=metadatas[i].get("document_id", "unknown"), 
            title=metadatas[i].get("title", "No title"),
            content_snippet=documents[i], # El chunk de texto
            similarity_score=max(0.0, score) # Evitar negativos
        )
        formatted_results.append(item)

    count_results = len(formatted_results)
    logger.info(f"Búsqueda finalizada. Se encontraron {count_results} fragmentos relevantes.")

    return {"results": formatted_results}


@app.post("/ask", response_model=AskQuestionResponse)
def ask(request: AskQuestionRequest):
    
    logger.info(f"--> Pregunta recibida: '{request.question}'")

    question = request.question
    
    # Generar embedding de la pregunta
    try:
        query_vector = get_embedding(question)
    except Exception:
         raise HTTPException(status_code=500, detail="Error al procesar la pregunta.")

    # Buscar el contexto más relevante (Top 1 es suficiente para responder)
    search_results = search_chunks(query_vector, n_results=1)
    
    # Extraer datos de ChromaDB
    # Si no hay documentos cargados, search_results['ids'] estará vacío
    if not search_results['ids'][0]:
        return AskQuestionResponse(
            answer="No cuento con información suficiente para responder a esta consulta.",
            context_used="",
            similarity_score=0.0,
            grounded=False
        )

    # Obtenemos el mejor candidato
    best_snippet = search_results['documents'][0][0]
    distance = search_results['distances'][0][0]
    # Convertir distancia a similitud (0 a 1)
    similarity = max(0.0, 1 - distance)

    # Validación de "Contexto Suficiente"
    if similarity < 0.5:
        logger.warning(f"Similitud baja ({similarity:.4f}). Rechazando respuesta por falta de contexto.")
        return AskQuestionResponse(
            answer="No cuento con información suficiente para responder a esta consulta.",
            context_used=best_snippet, # Mostramos qué encontró aunque no sirva (Transparencia)
            similarity_score=similarity,
            grounded=False
        )

    # Generar respuesta con LLM
    llm_response = generate_answer_with_context(question, best_snippet)
    
    if llm_response is None:
        raise HTTPException(status_code=500, detail="El servicio externo no pudo procesar la solicitud.")

    # Verificar si el LLM usó el mensaje de fallback
    fallback_msg = "No cuento con información suficiente para responder a esta consulta."
    is_grounded = fallback_msg not in llm_response and "No puedo responder" not in llm_response

    logger.info(f"Respuesta generada. Score: {similarity:.4f} | Grounded: {is_grounded}")
    
    if not is_grounded:
        logger.warning("El modelo indicó que no tenía información suficiente.")

    return AskQuestionResponse(
        answer=llm_response,
        context_used=best_snippet,
        similarity_score=similarity,
        grounded=is_grounded
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True, reload_excludes=["logs/*"])