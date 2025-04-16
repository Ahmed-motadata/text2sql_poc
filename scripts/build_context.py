import os
import sys
import json
from typing import Dict, Any, List

# Add the parent directory to the path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required functions from similar_tables and similar_columns
from scripts.similar_tables import get_similar_tables_from_user_query
from scripts.similar_columns import get_similar_columns_from_user_query
from core.helper import get_token_count_for_text
from settings.settings import DEFAULT_CONTEXT_SQL_PROMPT  # <-- Import the context SQL prompt
from base.prompts import _CONTEXT_TEMPLATE

# Set your default methods here
DEFAULT_TABLE_METHOD = "llm"
DEFAULT_COLUMN_METHOD = "llm" 

def get_table_method():
    return DEFAULT_TABLE_METHOD

def get_column_method():
    return DEFAULT_COLUMN_METHOD

def build_llm_context(user_query: str, column_search_method: str = DEFAULT_COLUMN_METHOD, table_search_method: str = DEFAULT_TABLE_METHOD) -> Dict[str, Any]:
    """
    Build context for LLM by retrieving similar tables and columns based on user query.
    
    Args:
        user_query (str): The user's query
        column_search_method (str): Method to use for column search - "vector" or "llm" (default: get_column_method())
        table_search_method (str): Method to use for table search - "vector" or "llm" (default: get_table_method())
        
    Returns:
        Dict containing vector_token_count, llm_token_usage, and llm_context
    """

    
    # Validate input parameters
    if column_search_method.lower() not in ["vector", "llm"]:
        raise ValueError("column_search_method must be either 'vector' or 'llm'")
    if table_search_method.lower() not in ["vector", "llm"]:
        raise ValueError("table_search_method must be either 'vector' or 'llm'")
    
    # Calculate token count of user query for vector search
    vector_token_count = get_token_count_for_text(user_query)
    
    # Initialize token usage counters
    llm_token_usage = 0
    
    # Step 1: Get similar tables based on user query using the specified method
    tables_result = get_similar_tables_from_user_query(user_query, method=table_search_method)
    
    # The function returns a list with a single item, so we get the first item
    table_info = tables_result[0]
    
    # Update token usage if table search used LLM
    if table_search_method.lower() == "llm":
        llm_token_usage += table_info.get("tokens_used_to_build", 0)
    
    # Extract table dictionary from the result
    tables_dict = table_info.get("tables", {})
    
    # Convert to list format for the column search function
    table_list = []
    for table_key, table_data in tables_dict.items():
        # Convert custom objects to dict for JSON serialization
        if hasattr(table_data, "model_dump"):
            table_list.append(table_data.model_dump())
        elif hasattr(table_data, "__dict__"):
            table_list.append(vars(table_data))
        elif isinstance(table_data, dict):
            table_list.append(table_data)
        else:
            # Fallback: try str conversion
            table_list.append(str(table_data))
    
    # Step 2: Get similar columns for the retrieved tables
    columns_result = get_similar_columns_from_user_query(user_query, table_list, method=column_search_method)
    
    # Get the first item from the result (similar to tables)
    column_info = columns_result[0] if isinstance(columns_result, list) else columns_result
    
    # Update token usage if column search used LLM
    if column_search_method.lower() == "llm":
        llm_token_usage += column_info.get("tokens_used_to_build", 0)
    
    # Step 4: Create the result structure based on search methods
    schema_context = format_schema_context({
        'tables': table_info.get('tables', {}),
        'columns': column_info.get('tables', {})
    })
    prompt = _CONTEXT_TEMPLATE.format(schema=schema_context, user_query=user_query)
    result = {
        "vector_token_count": vector_token_count if "vector" in [column_search_method.lower(), table_search_method.lower()] else 0,
        "llm_token_usage": llm_token_usage,
        "llm_context": prompt
    }
    return result

def format_schema_context(result):
    tables = result.get('tables', {})
    columns = result.get('columns', {})
    lines = []
    lines.append("")
    lines.append(f"Here is the relevant ({len(tables)}) tables and their columns:")
    for t_idx, (table_key, table_info) in enumerate(tables.items(), 1):
        lines.append(f'<table{t_idx}_start>')
        table_name = getattr(table_info, 'name', None) or (table_info['name'] if isinstance(table_info, dict) else str(table_info))
        table_desc = getattr(table_info, 'description', None) or (table_info['description'] if isinstance(table_info, dict) else "")
        lines.append(f'    "name" : "{table_name}"')
        lines.append(f'    "description" : "{table_desc}"')
        table_columns = columns.get(table_key, {})
        if hasattr(table_columns, 'columns'):
            col_list = table_columns.columns
        elif isinstance(table_columns, dict) and 'columns' in table_columns:
            col_list = table_columns['columns']
        else:
            col_list = []
        lines.append(f'\n        Here are the relevant ({len(col_list)}) columns of this table')
        for c_idx, col in enumerate(col_list, 1):
            col_name = getattr(col, 'name', None) or (col['name'] if isinstance(col, dict) else str(col))
            col_datatype = getattr(col, 'data_type', None) or (col.get('data_type') if isinstance(col, dict) else None)
            col_desc = ""
            if hasattr(col, 'description'):
                col_desc = col.description
            elif isinstance(col, dict) and 'description' in col:
                col_desc = col['description']
            if isinstance(col_desc, str) and ("COLUMN_NAME:" in col_desc or "TABLE:" in col_desc):
                desc_lines = [l.strip() for l in col_desc.split('\n') if l.strip()]
                if len(desc_lines) > 1:
                    col_desc = desc_lines[-1]
            datatype_str = f'({col_datatype})' if col_datatype else ''
            lines.append(f'        "{c_idx}" : "{col_name}"{datatype_str} : "{col_desc}"')
        lines.append(f'<table{t_idx}_end>')
    return "\n".join(lines)

def get_llm_context(user_query: str):
    result = build_llm_context(user_query, get_column_method(), get_table_method())
    schema_context = format_schema_context(result)
    prompt = _CONTEXT_TEMPLATE.format(schema=schema_context, user_query=user_query)
    llm_context = prompt
    return llm_context

# Test the function when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build LLM context based on user query")
    parser.add_argument("user_query", nargs="?", help="The user's query")
    
    args = parser.parse_args()
    
    # Prompt for user input if no query is provided via command line
    if args.user_query is None:
        args.user_query = input("Please enter your query: ")
    
    try:
        llm_context = get_llm_context(args.user_query)
        # print(result['llm_token_usage'])
        # print(result['vector_token_count'])
        print(llm_context)
    except Exception as e:
        print(f"Error: {str(e)}")