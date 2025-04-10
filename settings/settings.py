"""
Configuration settings for Text2SQL application.

This module provides centralized settings for:
- Default LLM model for general use
- Query generation LLM model
- Embedding models for vectorization
- Vector store configuration
- Default prompts
"""

import os
import sys
from enum import Enum
from typing import Dict, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models to make them available
from base.chat_models import (
    chat_gpt4o,
    chat_gpt4,
    chat_gpt_o3mini,
    chat_groq_llama,
    chat_groq_llama_70b,
    chat_groq_deepseek,
    chat_gemini_flash,
    chat_gemini_flash_lite
)

# Import embedding model functions instead of instances
from base.embed_models import (
    get_openai_embed_ada,
    get_openai_embed_3_small,
    get_openai_embed_3_large,
    get_jina_embed_base,
    get_jina_embed_clip,
    get_jina_embed_small
)

# Import prompts
from base.prompts import (
    SQL_GENERATION_PROMPT,
    SQL_GENERATION_WITH_EXAMPLES_PROMPT,
    SQL_EXPLANATION_PROMPT,
    SQL_DEBUGGING_PROMPT,
    SQL_OPTIMIZATION_PROMPT,
    ALL_PROMPTS
)

# Define model enumerations for type checking and autocompletion
class LLMModel(str, Enum):
    """Available LLM models"""
    GPT4O = "gpt-4o"
    GPT4 = "gpt-4"
    GPT_O3MINI = "gpt-o3-mini"
    GROQ_LLAMA = "llama-3.3-70b-versatile"
    GROQ_LLAMA_70B = "llama3-70b-8192"
    GROQ_DEEPSEEK = "deepseek-r1-distill-llama-70b"
    GEMINI_FLASH = "gemini-2.0-flash-001"
    GEMINI_FLASH_LITE = "gemini-2.0-flash-lite-001"

class EmbeddingModel(str, Enum):
    """Available embedding models"""
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    JINA_BASE = "jina-embeddings-v2-base-en"
    JINA_CLIP = "jina-clip-v2"
    JINA_SMALL = "jina-embeddings-v2-small-en"

# =========== DEFAULT SETTINGS ===========

# Default LLM model (used for general purposes)
DEFAULT_LLM = chat_gemini_flash

# Default query generation LLM model (used specifically for generating SQL queries)
DEFAULT_QUERY_LLM = chat_gemini_flash

# Default embedding model function (used for vectorizing text)
DEFAULT_EMBEDDING_MODEL_FUNC = get_jina_embed_base

# Default prompt for SQL generation
DEFAULT_SQL_PROMPT = SQL_GENERATION_PROMPT

# =========== DATABASE SETTINGS ===========

# Database file paths for token counting functionality
DATABASE_SETTINGS = {
    "input_db_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "database", "db_metadata.json"),
    "output_paths": {
        "column_token_count": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "database", "get_column_token_count.json"),
        "table_token_count": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        "database", "get_table_token_count.json"),
        "token_count": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "database", "get_token_count.json"),
        "processed_db": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "database", "processed_db.json")
    }
}

# =========== VECTOR STORE SETTINGS ===========

# Default Qdrant vector store settings
VECTOR_STORE_SETTINGS = {
    "host": os.getenv("QDRANT_HOST", "localhost"),
    "port": int(os.getenv("QDRANT_PORT", 6333)),
    "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", 6334)) if os.getenv("QDRANT_GRPC_PORT") else None,
    "api_key": os.getenv("QDRANT_API_KEY"),
    "https": os.getenv("QDRANT_HTTPS", "False").lower() == "true",
    "prefer_grpc": os.getenv("QDRANT_PREFER_GRPC", "False").lower() == "true",
    "timeout": float(os.getenv("QDRANT_TIMEOUT", 10.0))
}

# Default collection settings for schema metadata
DEFAULT_COLLECTION_NAME = "text2sql_schema"
DEFAULT_VECTOR_SIZE = 768  # Default for jina-embeddings-v2-base-en
DEFAULT_DISTANCE = "cosine"

# =========== MODEL MAPPINGS ===========

# Mapping from model enum to actual model instances
LLM_MODEL_MAPPING = {
    LLMModel.GPT4O: chat_gpt4o,
    LLMModel.GPT4: chat_gpt4,
    LLMModel.GPT_O3MINI: chat_gpt_o3mini,
    LLMModel.GROQ_LLAMA: chat_groq_llama,
    LLMModel.GROQ_LLAMA_70B: chat_groq_llama_70b,
    LLMModel.GROQ_DEEPSEEK: chat_groq_deepseek,
    LLMModel.GEMINI_FLASH: chat_gemini_flash,
    LLMModel.GEMINI_FLASH_LITE: chat_gemini_flash_lite,
}

# Mapping from model enum to embedding model functions
EMBEDDING_MODEL_MAPPING = {
    EmbeddingModel.OPENAI_ADA: get_openai_embed_ada,
    EmbeddingModel.OPENAI_3_SMALL: get_openai_embed_3_small,
    EmbeddingModel.OPENAI_3_LARGE: get_openai_embed_3_large,
    EmbeddingModel.JINA_BASE: get_jina_embed_base,
    EmbeddingModel.JINA_CLIP: get_jina_embed_clip,
    EmbeddingModel.JINA_SMALL: get_jina_embed_small,
}

# =========== HELPER FUNCTIONS ===========

def get_llm_model(model_name: Optional[str] = None):
    """
    Get the appropriate LLM model based on the model name.
    
    Args:
        model_name: The name of the model to retrieve (must match an LLMModel enum value)
        
    Returns:
        The corresponding LLM model instance
    """
    if not model_name:
        return DEFAULT_LLM
        
    try:
        model_enum = LLMModel(model_name)
        return LLM_MODEL_MAPPING[model_enum]
    except (ValueError, KeyError):
        print(f"Warning: Model '{model_name}' not found. Using default model.")
        return DEFAULT_LLM

def get_query_llm(model_name: Optional[str] = None):
    """
    Get the appropriate query generation LLM model.
    
    Args:
        model_name: The name of the model to retrieve (must match an LLMModel enum value)
        
    Returns:
        The corresponding LLM model instance for query generation
    """
    if not model_name:
        return DEFAULT_QUERY_LLM
        
    try:
        model_enum = LLMModel(model_name)
        return LLM_MODEL_MAPPING[model_enum]
    except (ValueError, KeyError):
        print(f"Warning: Model '{model_name}' not found. Using default query model.")
        return DEFAULT_QUERY_LLM

def get_embedding_model(model_name: Optional[str] = None):
    """
    Get the appropriate embedding model.
    
    Args:
        model_name: The name of the model to retrieve (must match an EmbeddingModel enum value)
        
    Returns:
        The corresponding embedding model instance
    """
    if not model_name:
        return DEFAULT_EMBEDDING_MODEL_FUNC()
        
    try:
        model_enum = EmbeddingModel(model_name)
        model_func = EMBEDDING_MODEL_MAPPING[model_enum]
        return model_func()
    except (ValueError, KeyError):
        print(f"Warning: Embedding model '{model_name}' not found. Using default embedding model.")
        return DEFAULT_EMBEDDING_MODEL_FUNC()

def get_prompt(prompt_type: str = "generation"):
    """
    Get a prompt template by type.
    
    Args:
        prompt_type: Type of prompt to retrieve (matches keys in ALL_PROMPTS)
        
    Returns:
        The corresponding prompt template or default prompt if not found
    """
    if prompt_type in ALL_PROMPTS:
        return ALL_PROMPTS[prompt_type]
    print(f"Warning: Prompt type '{prompt_type}' not found. Using default SQL generation prompt.")
    return DEFAULT_SQL_PROMPT

def get_vector_store_config():
    """
    Get the vector store configuration parameters.
    
    Returns:
        Dictionary with vector store connection settings
    """
    return VECTOR_STORE_SETTINGS

def get_collection_config(collection_name=None):
    """
    Get collection configuration for vector store.
    
    Args:
        collection_name: Optional name of collection to use (defaults to DEFAULT_COLLECTION_NAME)
        
    Returns:
        Dictionary with collection configuration parameters
    """
    return {
        "name": collection_name or DEFAULT_COLLECTION_NAME,
        "vector_size": DEFAULT_VECTOR_SIZE,
        "distance": DEFAULT_DISTANCE
    }