from langchain_postgres import PGVector
import sys
import os
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.settings import DEFAULT_EMBEDDING_MODEL, PGVECTOR_CONNECTION_STRING, PGVECTOR_COLLECTION_NAME
from base.embed_models import JINA_EMBED_V3
 
def get_vector_store(
    collection_name: str = PGVECTOR_COLLECTION_NAME,
    connection_string: str = PGVECTOR_CONNECTION_STRING,
    embedding_model = DEFAULT_EMBEDDING_MODEL
) -> PGVector:
    return PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )
 
vector_store = get_vector_store()