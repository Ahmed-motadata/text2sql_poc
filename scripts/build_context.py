import os
import sys
from typing import Dict, Any
from pydantic import BaseModel, Field
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
 
from scripts.similar_tables import get_similar_tables_from_user_query
from scripts.similar_columns import get_similar_columns_from_user_query
from core.helper import get_token_count_for_text
from settings.settings import DEFAULT_CONTEXT_SQL_PROMPT
 
class LLMContext(BaseModel):
    vector_token_count: int = Field(0)
    llm_token_usage: int = Field(0)
    llm_context: Dict[str, Any] = Field(...)
 
 
 
def build_llm_context(user_query: str, column_search_method: str = "vector", table_search_method: str = "vector") -> Dict[str, Any]:
    """
    Build context for LLM by retrieving similar tables and columns based on user query.
 
    Args:
        user_query (str): The user's query.
        column_search_method (str): Method to use for column search - "vector" or "llm".
        table_search_method (str): Method to use for table search - "vector" or "llm".
 
    Returns:
        Dict containing vector_token_count, llm_token_usage, llm_context, user_query, and table/column docs.
    """
    # Validate input parameters
    if column_search_method.lower() not in ["vector", "llm"]:
        raise ValueError("column_search_method must be either 'vector' or 'llm'")
    if table_search_method.lower() not in ["vector", "llm"]:
        raise ValueError("table_search_method must be either 'vector' or 'llm'")
 
    # Step 1: Get similar tables
    tables_result = get_similar_tables_from_user_query(user_query, method=table_search_method)
    tables_dict = tables_result[0].get("tables", {})
 
    # Step 2: Prepare table list for column search
    table_list = [{"name": table_data.name, "description": table_data.description} for table_data in tables_dict.values()]
 
    # Step 3: Get similar columns
    columns_result = get_similar_columns_from_user_query(user_query, table_list, method=column_search_method)
    columns_dict = columns_result[0].get("tables", {})
    column_docs = ""
    for name, table_data in columns_dict.items():
        column_docs += f"\nColumns for Table: {name}\n"
        for col in table_data.columns:
            column_docs += f"  Column: {col.name}, Description: {col.description}\n"
 
    
    # Step 4: Build schema string as per your desired format
    schema = ""
    for idx, (table_name, table_data) in enumerate(columns_dict.items(), 1):
        schema += f"<table-{idx}_start>\n"
        table_name = table_data.name  # safer if name is stored here
 
        schema += f"table: {table_name}\n"
        schema += f"description : {table_data.description}\n"
        schema += f"{table_name} table columns is mentioned below\n"
        
        for col_idx, col in enumerate(table_data.columns, 1):
            col_desc = col.description.strip() if col.description and col.description.strip() else "No description available"
            schema += f"{col_idx}. {col.name} : {col_desc}\n"
        
            
        
        schema += f"<table-{idx}_end>\n\n"
 
 
 
        # Step 4: Build LLM context
        
    llm_context = DEFAULT_CONTEXT_SQL_PROMPT.template.format(
        schema=schema.strip(),
        user_query=user_query.strip()
    )
 
    
    vector_token_count = 0
    llm_token_usage = 0

    # Table search token usage
    table_token_count = tables_result[0].get("tokens_used_to_build", 0)
    if table_search_method.lower() == "vector":
        vector_token_count += table_token_count
    else:  # llm
        llm_token_usage += table_token_count

    # Column search token usage
    column_token_count = columns_result[0].get("tokens_used_to_build", 0)
    if column_search_method.lower() == "vector":
        vector_token_count += column_token_count
    else:  # llm
        llm_token_usage += column_token_count

    #print(llm_token_usage)    
 
    
    return {
        "llm_context": llm_context,
        "llm_token_usage": llm_token_usage,
        "vector_token_usage": vector_token_count,
        "total_token_usage": vector_token_count + llm_token_usage,
 
    }
 
 
 
result = build_llm_context("Show high priority tickets", column_search_method="llm", table_search_method="llm")
# print(result["llm_context"])
# print(result["vector_token_usage"])
# print(result["llm_token_usage"])
 