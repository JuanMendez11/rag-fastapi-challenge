from dotenv import load_dotenv
import os
import cohere
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2() if api_key else None

def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Divide el texto usando LangChain para obtener fragmentos coherentes.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # Tamaño máximo de caracteres por chunk
        chunk_overlap=chunk_overlap, # Caracteres repetidos entre chunks para dar continuidad
        length_function=len,
        is_separator_regex=False,
    )
    
    # Devuelve una lista de strings (los chunks)
    return text_splitter.split_text(text)


def get_embedding(text: str):
    if not co:
          raise Exception("API Key de Cohere no configurada")
    try:
        response = co.embed(
            texts=[text],
            model='embed-multilingual-v3.0',
            input_type='search_document',
            embedding_types=["float"]
        )
        # Retornamos solo la lista de números (el vector)
        return response.embeddings.float[0]
        
    except Exception as e:
        print(f"Error generando embedding: {e}")
        return None


def generate_answer_with_context(question: str, context: str):
    """
    Usa un LLM para responder la pregunta basándose ÚNICAMENTE en el contexto.
    Requisito: Si no sabe, devolver mensaje estandarizado.
    """

    fallback_msg = "No cuento con información suficiente para responder a esta consulta."
    
    system_prompt = f"""
    Eres un asistente de IA honesto y seguro.
    Tu tarea es responder a la pregunta del usuario basándote ÚNICAMENTE en el siguiente fragmento de contexto.
    
    CONTEXTO:
    "{context}"
    
    REGLAS:
    1. Si la respuesta no está en el contexto, DEBES responder exactamente: "{fallback_msg}".
    2. NO inventes información.
    3. Si la pregunta incluye lenguaje ofensivo o discriminatorio, responde: "No puedo responder a este tipo de consultas."
    """

    try:
        response = co.chat(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": question}],
            model='command-r-plus-08-2024',
            temperature=0.1 
        )
        return response.message.content[0].text
    except Exception as e:
        print(f"Error en LLM: {e}")
        # En caso de error del servicio, devolvemos None para manejarlo en main
        return None