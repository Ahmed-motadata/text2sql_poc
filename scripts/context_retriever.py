import os
import sys
import json
import logging
from typing import List, Dict, Any

# Add the parent directory to the path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retriever import Retriever
from settings.settings import DEFAULT_TABLE_FETCHING_LLM, DEFAULT_COLUMN_FETCHING_LLM, get_similar_tables_prompt, get_similar_columns_prompt, DATABASE_SETTINGS
from core.helper import get_token_count_for_text, get_table_token_count

# Define these classes here since they're not in core.helper
class ColumnInfo:
    """Class to represent column information"""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

class TableColumns:
    """Class to represent table columns information"""
    def __init__(self, name: str, description: str = "", columns: List[ColumnInfo] = None):
        self.name = name
        self.description = description
        self.columns = columns or []

def get_similar_tables_from_user_query(user_query: str, method: str = "vector") -> List[Dict[str, Any]]:
    """
    Retrieve similar tables based on a user query using either vector embeddings or LLM.

    Args:
        user_query (str): The user's query about tables
        method (str): Method to use - "vector" or "llm"

    Returns:
        List containing a dictionary with the method and retrieved tables or raw LLM output
    """
    if method.lower() not in ["vector", "llm"]:
        raise ValueError("Method must be either 'vector' or 'llm'")

    if method.lower() == "vector":
        result = _get_tables_vector_method(user_query)
    else:
        result = _get_tables_llm_method(user_query)

    return [result]

def get_similar_columns_from_user_query(user_query: str, table_list: List[Dict[str, Any]], method: str = "vector") -> List[Dict[str, Any]]:
    """
    Main entry point to retrieve columns from tables based on user query.
    
    Args:
        user_query (str): The user's query about columns
        table_list (List[Dict]): List of tables to retrieve columns for
        method (str): Method to use - "vector" or "llm"
        
    Returns:
        List containing a dictionary with the method and retrieved columns
    """
    if method.lower() not in ["vector", "llm"]:
        raise ValueError("Method must be either 'vector' or 'llm'")

    if method.lower() == "vector":
        result = _get_columns_vector_method(user_query, table_list)
    else:
        input_db_path = DATABASE_SETTINGS.get("input_db_path")
        result = _get_column_llm_method(user_query, input_db_path)

    return [result]

def _get_tables_vector_method(user_query: str) -> Dict[str, Any]:
    """
    Use vector embeddings to find similar tables.

    Args:
        user_query (str): The user's query

    Returns:
        Dict containing method, token count and retrieved tables
    """
    token_counts_user_query = get_token_count_for_text(user_query)
    k = 10
    threshold = 0.7
    retriever = Retriever(k, threshold, user_query)
    table_docs = retriever.get_table_doc()

    tables_list = []
    for idx, (doc, score) in enumerate(table_docs, start=1):
        name = doc.metadata.get('table_name') or ''
        if not name and 'Table:' in doc.page_content:
            name = doc.page_content.splitlines()[0].replace('Table:', '').strip()
        if not name:
            name = f"table_{idx}"

        description = doc.metadata.get('description', '')
        if not description:
            lines = doc.page_content.splitlines()[1:]
            description = next((line for line in lines if line.strip()), '')
        if not description:
            description = f"Table identified with similarity score: {score:.2f}"

        tables_list.append({"table": name, "description": description})

    return {"method": "vector", "tokens_used_to_build": token_counts_user_query, "tables": tables_list}

def _get_tables_llm_method(user_query: str) -> Dict[str, Any]:
    """
    Use an LLM to identify similar tables based on the user query.

    Builds a prompt from the schema metadata and user query, then returns prettified table list and token usage.

    Args:
        user_query (str): The user's query

    Returns:
        Dict containing method, prettified tables, and token usage
    """
    input_path = DATABASE_SETTINGS.get("input_db_path")
    if not isinstance(input_path, str):
        raise ValueError("CONFIG ERROR: 'input_db_path' must be a file path string")

    with open(input_path, 'r') as f:
        schema_json = json.load(f)

    db_list = schema_json if isinstance(schema_json, list) else [schema_json]
    get_table_token_count(schema_json, [])

    table_descriptions: Dict[str, str] = {}
    for db in db_list:
        for tbl in db.get('tables', []):
            table_descriptions[tbl.get('name', '')] = tbl.get('description', '')

    all_tables_info = ''
    for tbl_name, desc in table_descriptions.items():
        all_tables_info += f"- {tbl_name}: {desc}\n"

    prompt_template = get_similar_tables_prompt()
    prompt = prompt_template.format(
        all_tables_info=all_tables_info,
        user_query=user_query
    )

    # Track token count of the prompt for accurate input token usage
    prompt_token_count = get_token_count_for_text(prompt)

    llm = DEFAULT_TABLE_FETCHING_LLM
    raw_response = llm.invoke(prompt)
    content = getattr(raw_response, 'content', str(raw_response))

    prettified_tables = []
    for idx, line in enumerate(content.strip().split("\n"), start=1):
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                prettified_tables.append({"table": parts[0].strip(), "description": parts[1].strip()})

    # Extract token usage from the LLM response metadata
    input_tokens = None
    output_tokens = None
    total_tokens = None
    
    if hasattr(raw_response, 'response_metadata'):
        token_data = raw_response.response_metadata.get('token_usage', {})
        input_tokens = token_data.get('prompt_tokens', prompt_token_count)
        output_tokens = token_data.get('completion_tokens')
        total_tokens = token_data.get('total_tokens')
        
        # Fallback to calculating total if not provided
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

    return {
        "method": "llm",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens, 
        "total_tokens": total_tokens,
        "tables": prettified_tables,
        "tokens_used_to_build": total_tokens  # For compatibility with existing code
    }

def parse_table_list(raw_result: List[Dict[str, Any]]) -> List[str]:
    """
    Extract table names from the table retrieval output.
    Supports both vector-based and LLM-based output formats.

    If 'tables' is a dictionary (vector), extract keys.
    If 'tables' is a list of dicts with 'table' field, extract those names.
    """
    table_names = []

    if not raw_result:
        return table_names

    tables = raw_result[0].get("tables")

    if isinstance(tables, dict):  # Vector output case
        table_names = [table_info.get("table") for table_info in tables.values() if table_info.get("table")]

    elif isinstance(tables, list):  # LLM output case
        table_names = [entry.get("table") for entry in tables if "table" in entry]

    return table_names

def _get_columns_vector_method(user_query: str, table_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use vector embeddings to find similar columns in the specified tables.

    Args:
        user_query (str): The user's query about columns.
        table_list (List[Dict]): List of tables to search columns in.

    Returns:
        Dict containing method, token count and retrieved columns organized by table.
    """
    # Calculate the token count of the user query for tracking
    token_counts_user_query = get_token_count_for_text(user_query)
    
    # Set parameters for the vector retrieval - increased k and lowered threshold for more results
    k = 10  # Increased from 10 to 20 to retrieve more columns
    threshold = 0.5  # Lowered from 0.6 to 0.5 to capture more potentially relevant columns
    
    # Create a retriever instance with the required parameters
    retriever = Retriever(k, threshold, user_query)
    
    # Prepare table names - extract from table_list with robust handling
    table_names = []
    for table in table_list:
        # Try multiple fields that might contain the table name
        table_name = None
        if "name" in table:
            table_name = table["name"]
        elif "table" in table:
            table_name = table["table"]
        elif "table_name" in table:
            table_name = table["table_name"]
        
        if table_name and table_name not in table_names:
            table_names.append(table_name)
    
    # Store descriptions for each table for later use
    table_descriptions = {}
    for table in table_list:
        name = table.get("name") or table.get("table") or table.get("table_name")
        if name:
            table_descriptions[name] = table.get("description", "")
    
    # Initialize a dictionary to store column data for each table
    tables_columns_dict = {}
    
    # Get column documents for all tables at once - better performance and results
    column_docs = retriever.get_column_doc(table_names)
    
    # Process column documents and organize by table
    for doc, score in column_docs:
        # Get the column name and table name from the document's metadata
        column_name = doc.metadata.get('column_name')
        table_name = doc.metadata.get('table_name')
        
        if not column_name or not table_name:
            continue  # Skip if no column name or table name found
        
        # Get the column description from the document metadata or content
        description = doc.metadata.get('description', '')
        if not description and doc.page_content:
            description = doc.page_content
        
        # If no description was found, create a default description with score
        if not description:
            description = f"Column identified with similarity score: {score:.2f}"
        
        # Get data type if available
        data_type = doc.metadata.get('data_type', '')
        
        # If this table is not yet in our result dictionary, initialize it
        if table_name not in tables_columns_dict:
            tables_columns_dict[table_name] = []
        
        # Add this column to the table's column list if not already present
        column_exists = False
        for existing_col in tables_columns_dict[table_name]:
            if existing_col.name == column_name:
                column_exists = True
                break
        
        if not column_exists:
            # Create a ColumnInfo object with name, description, and data_type
            column_info = ColumnInfo(name=column_name, description=description)
            # Add data_type as an attribute
            setattr(column_info, 'data_type', data_type)
            # Add to the list
            tables_columns_dict[table_name].append(column_info)
    
    # For tables with no columns yet, try to add common/important columns
    for table_name in table_names:
        # Skip if we already have columns for this table
        if table_name in tables_columns_dict and tables_columns_dict[table_name]:
            continue
            
        # Load schema to find important columns
        try:
            input_db_path = DATABASE_SETTINGS.get("input_db_path")
            with open(input_db_path, "r") as f:
                schema_json = json.load(f)
            
            db_list = schema_json if isinstance(schema_json, list) else [schema_json]
            for db in db_list:
                for table in db.get('tables', []):
                    if table.get('name') == table_name and 'columns' in table:
                        # Include key columns like id, name, and foreign keys
                        tables_columns_dict[table_name] = []
                        for col in table['columns']:
                            col_name = col.get('name', '')
                            # Include primary keys, common joins, and descriptive fields
                            if (col_name.lower().endswith('id') or  # ID fields
                                col_name.lower() in ('id', 'name', 'description', 'title', 'subject', 'status', 'type', 'priority') or  # Common important fields
                                'date' in col_name.lower() or 'time' in col_name.lower()):  # Date/time fields
                                
                                column_info = ColumnInfo(
                                    name=col_name, 
                                    description=col.get('description', '')
                                )
                                setattr(column_info, 'data_type', col.get('data_type', ''))
                                tables_columns_dict[table_name].append(column_info)
        except Exception as e:
            logging.error(f"Error loading schema to find important columns: {str(e)}")
    
    # Create the final result structure with the method used and token count
    result = {
        "method": "vector",
        "tokens_used_to_build": token_counts_user_query,
        "tables": tables_columns_dict
    }
    
    return result

def _get_column_llm_method(user_query: str, input_db_path: str, column_count_threshold: int = 10) -> Dict[str, Any]:
    """
    Retrieves relevant columns for a list of tables from a user query using LLM method.
    
    Args:
        user_query (str): The user query to fetch relevant columns.
        input_db_path (str): The path to the schema file.
        column_count_threshold (int): The maximum number of columns to fetch for each table.
        
    Returns:
        dict: A dictionary containing table names as keys and lists of relevant column names as values.
    """
    try:
        # Load the database schema
        with open(input_db_path, "r") as f:
            schema_json = json.load(f)
        
        # Handle schema structure - it could be a list or a dictionary
        db_list = schema_json if isinstance(schema_json, list) else [schema_json]
        
        # Extract tables_info from the schema
        tables_info = []
        for db in db_list:
            if 'tables' in db and isinstance(db['tables'], list):
                tables_info.extend(db['tables'])
        
        # Create a more detailed schema representation with table and column info
        detailed_schema = []
        for table in tables_info:
            if isinstance(table, dict) and 'name' in table:
                table_info = f"{table['name']}: {table.get('description', '')}"
                
                # Add column information if available
                if 'columns' in table and isinstance(table['columns'], list):
                    column_details = []
                    for col in table['columns']:
                        if isinstance(col, dict) and 'name' in col:
                            col_type = f"({col.get('data_type', '')})" if 'data_type' in col else ""
                            col_desc = col.get('description', '')
                            column_details.append(f"  - {col['name']} {col_type}: {col_desc}")
                    
                    if column_details:
                        table_info += "\n  Columns:\n" + "\n".join(column_details)
                
                detailed_schema.append(table_info)
        
        tables_info_str = "\n\n".join(detailed_schema)
        
        # Create an enhanced prompt that explicitly requests key columns even if not directly mentioned
        enhanced_query = f"{user_query}. Include ID columns, primary keys, foreign keys, and columns needed for joining tables, as well as any columns that would be useful for filtering, sorting, or displaying results."
        
        # Construct LLM prompt using the imported prompt template
        prompt_template = get_similar_columns_prompt()
        prompt = prompt_template.format(
            all_tables_columns_info=tables_info_str,
            user_query=enhanced_query
        )
        
        # Track token count of the prompt for accurate input token usage
        prompt_token_count = get_token_count_for_text(prompt)
        
        # Call LLM with the prompt
        llm = DEFAULT_COLUMN_FETCHING_LLM
        raw_response = llm.invoke(prompt)
        response = getattr(raw_response, 'content', str(raw_response))
        
        # Parse the LLM response to extract column names for each table
        columns_for_tables = {}
        
        # Create ColumnInfo objects for each column
        for table in tables_info:
            if isinstance(table, dict) and 'name' in table:
                table_name = table['name']
                relevant_columns_raw = _parse_llm_columns(response, table_name)
                
                # Convert string column names to ColumnInfo objects
                column_objects = []
                for col_name in relevant_columns_raw:
                    if col_name.strip():  # Skip empty column names
                        # Try to find column metadata in schema
                        col_desc = ""
                        col_type = ""
                        if 'columns' in table and isinstance(table['columns'], list):
                            for col_def in table['columns']:
                                if isinstance(col_def, dict) and col_def.get('name') == col_name:
                                    col_desc = col_def.get('description', '')
                                    col_type = col_def.get('data_type', '')
                                    break
                        
                        # Create ColumnInfo object
                        col_info = ColumnInfo(name=col_name, description=col_desc)
                        setattr(col_info, 'data_type', col_type)
                        column_objects.append(col_info)
                
                if column_objects:
                    columns_for_tables[table_name] = column_objects
                else:
                    # If no columns found via LLM, include essential columns from schema
                    backup_columns = []
                    if 'columns' in table and isinstance(table['columns'], list):
                        essential_column_names = ['id', 'name', 'description', 'title', 'subject', 'status', 'priority', 'created', 'updated']
                        for col in table['columns']:
                            col_name = col.get('name', '')
                            if col_name.lower() in essential_column_names or col_name.lower().endswith('id'):
                                col_info = ColumnInfo(name=col_name, description=col.get('description', ''))
                                setattr(col_info, 'data_type', col.get('data_type', ''))
                                backup_columns.append(col_info)
                                
                    columns_for_tables[table_name] = backup_columns
        
        # Extract token usage from the LLM response metadata
        input_tokens = None
        output_tokens = None
        total_tokens = None
        
        if hasattr(raw_response, 'response_metadata'):
            token_data = raw_response.response_metadata.get('token_usage', {})
            input_tokens = token_data.get('prompt_tokens', prompt_token_count)
            output_tokens = token_data.get('completion_tokens')
            total_tokens = token_data.get('total_tokens')
            
            # Fallback to calculating total if not provided
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
        
        # Create the final result structure with the method used and token count details
        result = {
            "method": "llm",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tables": columns_for_tables,
            "tokens_used_to_build": total_tokens  # For compatibility with existing code
        }
        
        return result
    
    except Exception as e:
        logging.error(f"Error in _get_column_llm_method: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "method": "llm", 
            "tables": {}, 
            "tokens_used_to_build": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

def _parse_llm_columns(response: str, table_name: str) -> List[str]:
    """
    Parse the LLM response to extract the relevant columns for a given table.
    
    Args:
        response (str): The LLM response containing columns for each table.
        table_name (str): The name of the table for which columns need to be parsed.
        
    Returns:
        List[str]: A list of relevant column names for the given table.
    """
    columns = []
    
    # Check for exact table name matches in the response
    for line in response.splitlines():
        # Look for table_name| pattern which is the expected format
        if line.startswith(f"{table_name}|"):
            columns_part = line[len(table_name)+1:].strip()
            # Split by comma and clean up each column name
            column_names = [col.strip() for col in columns_part.split(',')]
            columns.extend([col for col in column_names if col])  # Filter out empty strings
            return columns
            
    # More flexible matching if the exact pattern wasn't found
    table_prefix = f"{table_name}:"
    table_prefix_alt = f"{table_name} -"
    
    for line in response.splitlines():
        if table_name.lower() in line.lower():
            if line.startswith(table_prefix) or line.startswith(table_prefix_alt) or '|' in line:
                # Try to extract columns after a separator
                for sep in [':', '-', '|']:
                    if sep in line:
                        cols_part = line.split(sep, 1)[1].strip()
                        column_names = [col.strip() for col in cols_part.split(',')]
                        columns.extend([col for col in column_names if col])  # Filter out empty strings
                        break
    
    return columns
