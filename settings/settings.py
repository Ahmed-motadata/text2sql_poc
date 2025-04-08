"""
Configuration settings for Text2SQL application.

This module provides centralized settings for:
- Default LLM model for general use
- Query generation LLM model
- Embedding models for vectorization
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

from base.embed_models import (
    openai_embed_ada,
    openai_embed_3_small,
    openai_embed_3_large,
    jina_embed_base,
    jina_embed_clip,
    jina_embed_small
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

# Default embedding model (used for vectorizing text)
DEFAULT_EMBEDDING_MODEL = jina_embed_base

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

EMBEDDING_MODEL_MAPPING = {
    EmbeddingModel.OPENAI_ADA: openai_embed_ada,
    EmbeddingModel.OPENAI_3_SMALL: openai_embed_3_small,
    EmbeddingModel.OPENAI_3_LARGE: openai_embed_3_large,
    EmbeddingModel.JINA_BASE: jina_embed_base,
    EmbeddingModel.JINA_CLIP: jina_embed_clip,
    EmbeddingModel.JINA_SMALL: jina_embed_small,
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
        return DEFAULT_EMBEDDING_MODEL
        
    try:
        model_enum = EmbeddingModel(model_name)
        return EMBEDDING_MODEL_MAPPING[model_enum]
    except (ValueError, KeyError):
        print(f"Warning: Embedding model '{model_name}' not found. Using default embedding model.")
        return DEFAULT_EMBEDDING_MODEL