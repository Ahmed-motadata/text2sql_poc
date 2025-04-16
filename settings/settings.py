import os
import sys
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.chat_models import *
from base.embed_models import *
from base.prompts import * 

DEFAULT_LLM = OPENAI_4O_MINI
DEFAULT_QUERY_LLM = OPENAI_4O_MINI
    
DEFAULT_TABLE_FETCHING_LLM = OPENAI_4O_MINI
DEFAULT_COLUMN_FETCHING_LLM = OPENAI_4O_MINI
DEFAULT_EMBEDDING_MODEL = JINA_EMBED_V3
 
DEFAULT_SQL_PROMPT = SQL_GENERATION_PROMPT
 
DEFAULT_SIMILAR_TABLES_PROMPT = SIMILAR_TABLES_RETRIEVAL_PROMPT
 
DEFAULT_SIMILAR_COLUMNS_PROMPT = SIMILAR_COLUMNS_RETRIEVAL_PROMPT
 
DEFAULT_CONTEXT_SQL_PROMPT = CONTEXT_SQL_PROMPT
 
# =========== DATABASE SETTINGS ===========
DATABASE_SETTINGS = {
    "input_db_path": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "database", "db_metadata.json"),
    "output_path": {
        "processed_db": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "database", "processed_db.json")
    }
}
 
PGVECTOR_CONNECTION_STRING = (
    f"postgresql+psycopg://{os.getenv('PGVECTOR_USER')}:{os.getenv('PGVECTOR_PASSWORD')}"
    f"@{os.getenv('PGVECTOR_HOST')}:{os.getenv('PGVECTOR_PORT')}/{os.getenv('PGVECTOR_DB')}"
)
 
# Default collection name for vector store
PGVECTOR_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION", "T2sql_v3")
DEFAULT_COLLECTION_NAME = "text2sql_schema"
DEFAULT_VECTOR_SIZE = os.getenv("DEFAULT_VECTOR_SIZE")
DEFAULT_DISTANCE = os.getenv("DEFAULT_DISTANCE")
 
 
def get_similar_tables_prompt(custom_template: Optional[str] = None):
    """
    Get the similar tables retrieval prompt.
    Args:
        custom_template: Optional custom template to use instead of default
    Returns:
        The corresponding prompt template for similar tables retrieval
    """
    if custom_template:
        from langchain.prompts import PromptTemplate
        return PromptTemplate(
            input_variables=["all_tables_info", "user_query"],
            template=custom_template
        )
    return DEFAULT_SIMILAR_TABLES_PROMPT
 
def get_similar_columns_prompt(custom_template: Optional[str] = None):
    """
    Get the similar columns retrieval prompt.
    Args:
        custom_template: Optional custom template to use instead of default
    Returns:
        The corresponding prompt template for similar columns retrieval
    """
    if custom_template:
        from langchain.prompts import PromptTemplate
        return PromptTemplate(
            input_variables=["all_columns_info", "user_query"],
            template=custom_template
        )
    return DEFAULT_SIMILAR_COLUMNS_PROMPT