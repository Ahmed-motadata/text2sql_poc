from langchain_postgres import PGVector
from langchain.vectorstores import VectorStore
from typing import List, Dict, Any, Optional
import sys
import os

# Add the parent directory to the path to allow imports from settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.settings import get_embedding_model, DEFAULT_COLLECTION_NAME, VECTOR_STORE_SETTINGS

# Initialize connection string
connection_string = f"postgresql+psycopg://{VECTOR_STORE_SETTINGS.get('user', 'langchain')}:{VECTOR_STORE_SETTINGS.get('password', 'langchain')}@{VECTOR_STORE_SETTINGS['host']}:{VECTOR_STORE_SETTINGS['port']}/langchain"

def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model = None
) -> VectorStore:
    """
    Create and return a PGVector store instance with the specified configuration.
    
    Args:
        collection_name: Name of the collection in the vector store
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