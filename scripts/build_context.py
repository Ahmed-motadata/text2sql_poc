import os
import sys
import json
from typing import Dict, Any, List

# Add the parent directory to the path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required functions from context_retriever instead of similar_tables and similar_columns
from scripts.context_retriever import get_similar_tables_from_user_query, get_similar_columns_from_user_query
from core.helper import get_token_count_for_text
from settings.settings import DEFAULT_CONTEXT_SQL_PROMPT  # <-- Import the context SQL prompt
from base.prompts import _CONTEXT_TEMPLATE

# Set your default methods here
DEFAULT_TABLE_METHOD = "llm"
DEFAULT_COLUMN_METHOD = "vector"  # Changed from "vector" to "llm"

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
        Dict containing vector_token_count, llm_token_usage details, and llm_context
    """
    
    # Validate input parameters
    if column_search_method.lower() not in ["vector", "llm"]:
        raise ValueError("column_search_method must be either 'vector' or 'llm'")
    if table_search_method.lower() not in ["vector", "llm"]:
        raise ValueError("table_search_method must be either 'vector' or 'llm'")
    
    # Calculate token count of user query for vector search
    vector_token_count = 0
    
    # Initialize token usage counters for LLM methods
    table_input_tokens = 0
    table_output_tokens = 0
    column_input_tokens = 0
    column_output_tokens = 0
    
    # Step 1: Get similar tables based on user query using the specified method
    tables_result = get_similar_tables_from_user_query(user_query, method=table_search_method)
    
    # The function returns a list with a single item, so we get the first item
    table_info = tables_result[0]
    
    # Update token usage based on the method used
    if table_search_method.lower() == "vector":
        vector_token_count += table_info.get("tokens_used_to_build", 0)
    else:  # LLM method
        table_input_tokens = table_info.get("input_tokens", 0)
        table_output_tokens = table_info.get("output_tokens", 0)
    
    # Extract tables from the result - could be a list or dictionary
    tables_data = table_info.get("tables", [])
    
    # Convert to list format for the column search function
    table_list = []
    
    # Handle tables_data as either a list or a dictionary
    if isinstance(tables_data, list):
        # List format (new format from context_retriever.py)
        for i, table_data in enumerate(tables_data):
            # Process table data to ensure it has the expected structure with "name" field
            table_entry = {}
            
            # If table_data has table field but not name, create name from table field
            if "table" in table_data and "name" not in table_data:
                table_entry["name"] = table_data["table"]
            else:
                table_entry.update(table_data)
                
            # Ensure description is present
            if "description" not in table_entry and "description" in table_data:
                table_entry["description"] = table_data["description"]
            elif "description" not in table_entry:
                table_entry["description"] = ""
            
            # Ensure we have a name
            if "name" not in table_entry and "table" in table_entry:
                table_entry["name"] = table_entry["table"]
            elif "name" not in table_entry:
                table_entry["name"] = f"table_{i + 1}"
            
            table_list.append(table_entry)
    else:
        # Dictionary format (old format)
        for table_key, table_data in tables_data.items():
            # Process table data to ensure it has the expected structure with "name" field
            table_entry = {}
            
            # Handle different table data formats
            if isinstance(table_data, dict):
                # If table_data has table_name but not name, create name from table_name
                if "table_name" in table_data and "name" not in table_data:
                    table_entry["name"] = table_data["table_name"]
                else:
                    table_entry.update(table_data)
                    
                # Ensure description is present
                if "description" not in table_entry and "description" in table_data:
                    table_entry["description"] = table_data["description"]
                elif "description" not in table_entry:
                    table_entry["description"] = ""
            elif hasattr(table_data, "__dict__"):
                # If it's an object, convert it to dict
                table_dict = vars(table_data)
                # Fix name if needed
                if hasattr(table_data, "table_name") and not hasattr(table_data, "name"):
                    table_entry["name"] = table_data.table_name
                else:
                    table_entry.update(table_dict)
            else:
                # Fallback case: create a minimal entry with a name
                table_entry["name"] = str(table_data)
                table_entry["description"] = ""
            
            # Ensure we have a name
            if "name" not in table_entry and "table_name" in table_entry:
                table_entry["name"] = table_entry["table_name"]
            elif "name" not in table_entry:
                table_entry["name"] = f"table_{len(table_list) + 1}"
            
            table_list.append(table_entry)
    
    # Step 2: Get similar columns for the retrieved tables
    columns_result = get_similar_columns_from_user_query(user_query, table_list, method=column_search_method)
    
    # Get the first item from the result (similar to tables)
    column_info = columns_result[0] if isinstance(columns_result, list) else columns_result
    
    # Update token usage based on the method used for column retrieval
    if column_search_method.lower() == "vector":
        vector_token_count += column_info.get("tokens_used_to_build", 0)
    else:  # LLM method
        column_input_tokens = column_info.get("input_tokens", 0)
        column_output_tokens = column_info.get("output_tokens", 0)
    
    # Calculate total LLM token usage for compatibility with existing code
    total_llm_token_usage = table_input_tokens + table_output_tokens + column_input_tokens + column_output_tokens
    
    # Step 4: Create the result structure based on search methods
    schema_context = format_schema_context({
        'tables': tables_data,
        'columns': column_info.get('tables', {})
    })
    prompt = _CONTEXT_TEMPLATE.format(schema=schema_context, user_query=user_query)
    
    # Create comprehensive token usage details
    token_usage = {
        "table": {
            "input_tokens": table_input_tokens,
            "output_tokens": table_output_tokens,
            "total_tokens": table_input_tokens + table_output_tokens
        },
        "column": {
            "input_tokens": column_input_tokens,
            "output_tokens": column_output_tokens,
            "total_tokens": column_input_tokens + column_output_tokens
        },
        "total": {
            "input_tokens": table_input_tokens + column_input_tokens,
            "output_tokens": table_output_tokens + column_output_tokens,
            "total_tokens": total_llm_token_usage
        }
    }
    
    result = {
        "vector_token_count": vector_token_count,
        "llm_token_usage": total_llm_token_usage,  # For backward compatibility
        "token_usage": token_usage,  # New detailed token usage information
        "llm_context": prompt,
        "column_info": column_info
    }
    return result

def format_schema_context(result):
    tables = result.get('tables', {})
    columns = result.get('columns', {})
    lines = []
    lines.append("")
    
    # Convert tables to a more consistent format for processing
    processed_tables = []
    
    if isinstance(tables, dict):
        # Dictionary format (old format)
        for key, value in tables.items():
            if isinstance(value, dict) and "table_name" in value:
                # Handle the vector method output format
                processed_tables.append({
                    "key": key,
                    "name": value.get("table_name", key),
                    "description": value.get("description", "")
                })
            elif isinstance(value, dict) and "name" in value:
                # Already in the right format
                processed_tables.append({
                    "key": key,
                    "name": value.get("name", key),
                    "description": value.get("description", "")
                })
            else:
                # Fallback
                processed_tables.append({
                    "key": key,
                    "name": key,
                    "description": ""
                })
    elif isinstance(tables, list):
        # Handle list format (new format from context_retriever.py)
        for idx, table in enumerate(tables):
            if isinstance(table, dict):
                if "table" in table:
                    # New format with 'table' field
                    processed_tables.append({
                        "key": f"table_{idx + 1}",
                        "name": table.get("table"),
                        "description": table.get("description", "")
                    })
                elif "table_name" in table:
                    # Old format with 'table_name' field
                    processed_tables.append({
                        "key": f"table_{idx + 1}",
                        "name": table.get("table_name"),
                        "description": table.get("description", "")
                    })
                elif "name" in table:
                    # Format with 'name' field
                    processed_tables.append({
                        "key": f"table_{idx + 1}",
                        "name": table.get("name"),
                        "description": table.get("description", "")
                    })
    
    # Now add the line with the correct table count
    lines.append(f"Here is the relevant ({len(processed_tables)}) tables and their columns:")
    
    for t_idx, table in enumerate(processed_tables, 1):
        lines.append(f'<table{t_idx}_start>')
        lines.append(f'    "name" : "{table["name"]}"')
        lines.append(f'    "description" : "{table["description"]}"')
        
        # Handle columns - we need to match by table name, not key
        table_name = table["name"]
        col_list = []
        
        # Try to find columns for this table
        for key, cols in columns.items():
            # Extract the table name from the columns dict key
            if key == table_name or key == f"table_{t_idx}" or (isinstance(key, str) and table_name in key):
                if hasattr(cols, 'columns'):
                    col_list = cols.columns
                elif isinstance(cols, dict) and 'columns' in cols:
                    col_list = cols['columns']
                elif isinstance(cols, list):
                    col_list = cols
                break
        
        lines.append(f'\n        Here are the relevant ({len(col_list)}) columns of this table')
        
        for c_idx, col in enumerate(col_list, 1):
            col_name = ""
            col_datatype = ""
            col_desc = ""
            
            if hasattr(col, 'name'):
                col_name = col.name
                col_datatype = getattr(col, 'data_type', "")
                col_desc = getattr(col, 'description', "")
            elif isinstance(col, dict):
                col_name = col.get('name', "")
                col_datatype = col.get('data_type', "")
                col_desc = col.get('description', "")
            elif isinstance(col, str):
                col_name = col
            
            if isinstance(col_desc, str) and ("COLUMN_NAME:" in col_desc or "TABLE:" in col_desc):
                desc_lines = [l.strip() for l in col_desc.split('\n') if l.strip()]
                if len(desc_lines) > 1:
                    col_desc = desc_lines[-1]
            
            datatype_str = f'({col_datatype})' if col_datatype else ''
            lines.append(f'        "{c_idx}" : "{col_name}"{datatype_str} : "{col_desc}"')
        
        lines.append(f'<table{t_idx}_end>')
    
    return "\n".join(lines)

def get_llm_context(user_query: str):
    """Get the formatted context for the LLM based on the user query"""
    # Get the result from build_llm_context
    result = build_llm_context(user_query, get_column_method(), get_table_method())
    
    # Extract the tables and columns directly from the original source
    # This bypasses any transformations that might be happening in build_llm_context
    tables_result = get_similar_tables_from_user_query(user_query, method=get_table_method())
    table_info = tables_result[0]
    tables_data = table_info.get('tables', {})
    
    # Get columns data from the result
    column_info = result.get('column_info', {})
    columns_data = column_info.get('tables', {}) if column_info else {}
    
    # Format the schema context with the direct table and column data
    schema_context = format_schema_context({
        'tables': tables_data,
        'columns': columns_data
    })
    
    # Format the full prompt
    prompt = _CONTEXT_TEMPLATE.format(schema=schema_context, user_query=user_query)
    return prompt

# Test the function when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build LLM context based on user query")
    parser.add_argument("user_query", nargs="?", help="The user's query")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Prompt for user input if no query is provided via command line
    if args.user_query is None:
        args.user_query = input("Please enter your query: ")
    
    try:
        # Only show debug info if --debug flag is provided
        if args.debug:
            tables_result = get_similar_tables_from_user_query(args.user_query, method=get_table_method())
            print("\n--- DEBUG: Table Retrieval Results ---")
            print(f"Method used: {get_table_method()}")
            print(f"Tables found: {len(tables_result[0].get('tables', {}))}")
            print(f"Tables data: {tables_result[0].get('tables', {})}")
            print("-----------------------------------\n")
        
        # Get full context
        llm_context = get_llm_context(args.user_query)
        print(llm_context)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()