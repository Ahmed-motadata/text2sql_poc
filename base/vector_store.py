from langchain_postgres import PGVector
import sys
import os

# Add the parent directory to the path to allow imports from settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.settings import get_embedding_model, PGVECTOR_CONNECTION_STRING, PGVECTOR_COLLECTION_NAME
# Use the correct import path for embed_models
from base.embed_models import get_jina_embed_base

def get_vector_store(
    collection_name: str = PGVECTOR_COLLECTION_NAME,
    connection_string: str = PGVECTOR_CONNECTION_STRING,
    embedding_model = None
) -> PGVector:
    """
    Create and return a PGVector store instance with the specified configuration.
    
    Args:
        collection_name: Name of the collection in the vector store
        connection_string: PostgreSQL connection string
        embedding_model: Embedding model to use (if None, uses default from settings)
        
    Returns:
        An initialized PGVector instance
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()
    
    return PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )

# Create a default vector store instance for direct import
vector_store = get_vector_store()
