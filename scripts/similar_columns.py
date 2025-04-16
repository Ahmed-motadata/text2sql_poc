import os
import sys
import json
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Add the parent directory to the path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retriever import Retriever
from settings.settings import DEFAULT_COLUMN_FETCHING_LLM, get_similar_columns_prompt
from core.helper import get_token_count_for_text, get_table_token_count, get_column_token_count
from base.vector_store import vector_store

# For structured output parsing
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field as LCField
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Define Pydantic models for structured output
class ColumnInfo(BaseModel):
    name: str
    description: str

class TableColumns(BaseModel):
    name: str
    description: str
    columns: List[ColumnInfo]

class ColumnsResult(BaseModel):
    method: str = Field(..., description="Either 'vector' or 'llm'")
    tokens_used_to_build: int = Field(..., description="Token count of user query if vector, total token utilization if llm")
    tables: Dict[str, TableColumns] = Field(..., description="Dictionary of tables with their columns")

# LangChain compatible Pydantic models for structured output
class LCColumnInfo(LCBaseModel):
    name: str = LCField(description="The name of the column")
    description: str = LCField(description="A description of the column")

class LCTableColumns(LCBaseModel):
    columns: Dict[str, LCColumnInfo] = LCField(description="Dictionary of columns in the table")

class LCColumnsResult(LCBaseModel):
    tables: Dict[str, LCTableColumns] = LCField(description="Dictionary of tables with their columns")

# Helper function to suppress unwanted debug output
def suppress_output(func):
    """Decorator to suppress output during function execution"""
    def wrapper(*args, **kwargs):
        with io.StringIO() as buf, redirect_stdout(buf):
            result = func(*args, **kwargs)
        return result
    return wrapper

def get_similar_columns_from_user_query(user_query: str, table_list: List[Dict[str, Any]], method: str = "vector") -> List[Dict[str, Any]]:
    """
    Retrieve similar columns from specified tables based on a user query using either vector embeddings or LLM.
    
    Args:
        user_query (str): The user's query about columns
        table_list (List[Dict]): List of tables to search columns in, each with 'name' and 'description'
        method (str): Method to use - "vector" or "llm"
        
    Returns:
        List containing a dictionary with the method, token usage, and retrieved columns
    """
    if method.lower() not in ["vector", "llm"]:
        raise ValueError("Method must be either 'vector' or 'llm'")
    
    # Use the suppression decorator to prevent debug output
    if method.lower() == "vector":
        result = suppress_output(_get_columns_vector_method)(user_query, table_list)
    else:  # method == "llm"
        result = suppress_output(_get_columns_llm_method)(user_query, table_list)
    
    # Convert to list format as requested
    return [result]

def _get_columns_vector_method(user_query: str, table_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use vector embeddings to find similar columns in the specified tables.
    
    Args:
        user_query (str): The user's query
        table_list (List[Dict]): List of tables to search columns in
        
    Returns:
        Dict containing method, token count and retrieved columns organized by table
    """
    # Calculate token count of user query
    token_counts_user_query = get_token_count_for_text(user_query)
    
    # Set up retriever with appropriate parameters
    k = 10  # Number of results to retrieve per table
    threshold = 0.7  # Similarity threshold for filtering results
    
    # Create a retriever instance
    retriever = Retriever(k, threshold, user_query)
    
    # Extract table names from table_list
    table_names = [table['name'] for table in table_list if 'name' in table]
    
    # Create a mapping of table names to their descriptions
    table_descriptions = {table['name']: table.get('description', '') 
                         for table in table_list if 'name' in table}
    
    # Get column documents for each table
    tables_columns_dict = {}
    
    for idx, table_name in enumerate(table_names):
        # Get column documents for this table
        column_docs = retriever.get_column_doc([table_name])
        
        columns_list = []
        # Process each column document
        for col_idx, (doc, score) in enumerate(column_docs):
            # Get column name from metadata
            column_name = doc.metadata.get('column_name')
            if not column_name:
                continue
                
            # Get column description from metadata or content
            description = doc.metadata.get('description', '')
            if not description and doc.page_content:
                description = doc.page_content
            
            # If no description was found, use a default
            if not description:
                description = f"Column identified with similarity score: {score:.2f}"
                
            # Add to columns list
            columns_list.append(ColumnInfo(
                name=column_name,
                description=description
            ))
        
        # Add table with its columns to the result dictionary
        if columns_list:
            tables_columns_dict[f"table_{idx+1}"] = TableColumns(
                name=table_name,
                description=table_descriptions.get(table_name, ''),
                columns=columns_list
            )
    
    # Create the result structure
    result = {
        "method": "vector",
        "tokens_used_to_build": token_counts_user_query,
        "tables": tables_columns_dict
    }
    
    return result

def _get_columns_llm_method(user_query: str, table_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use an LLM to identify relevant columns in the specified tables based on the user query.
    
    Args:
        user_query (str): The user's query
        table_list (List[Dict]): List of tables to search columns in
        
    Returns:
        Dict containing method, token count and retrieved columns organized by table
    """
    # Load database metadata
    from settings.settings import DATABASE_SETTINGS
    input_db_path = DATABASE_SETTINGS["input_db_path"]
    processed_db_path = DATABASE_SETTINGS["output_path"]["processed_db"]
    
    try:
        # Try to load the processed DB first as it has more structured data
        with open(processed_db_path, "r") as f:
            processed_db = json.load(f)
            initial_db = [processed_db]  # Wrap in list for compatibility
    except Exception as e:
        # Silently fall back to raw DB
        try:
            with open(input_db_path, "r") as f:
                initial_db = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading database metadata: {str(e)}")
    
    # Extract table names from table_list
    table_names = [table['name'] for table in table_list if 'name' in table]
    
    if not table_names:
        raise ValueError("No valid table names provided in table_list")
    
    # Create input for get_column_token_count
    column_input = [{"table": table_name, "column": None} for table_name in table_names]
    
    # Get column information for the specified tables
    columns_data = get_column_token_count(initial_db, column_input)
    
    # Create a mapping of table names to their column information
    table_to_columns = {}
    for table_info in columns_data:
        table_name = table_info.get("table_name", "")
        if table_name:
            table_to_columns[table_name] = table_info.get("columns", [])
    
    # Create a mapping of table names to their descriptions
    table_descriptions = {table['name']: table.get('description', '') 
                         for table in table_list if 'name' in table}
    
    # Try to load all column descriptions directly from the processed_db
    # This provides a more reliable source of column descriptions
    full_column_descriptions = {}
    for db in initial_db:
        for table in db.get("tables", []):
            table_name = table.get("name", "")
            if table_name:
                if table_name not in full_column_descriptions:
                    full_column_descriptions[table_name] = {}
                
                for column in table.get("columns", []):
                    col_name = column.get("name", "")
                    if col_name:
                        col_desc = column.get("description", "")
                        full_column_descriptions[table_name][col_name.lower()] = {
                            "description": col_desc,
                            "data_type": column.get("data_type", ""),
                            "correct_name": col_name
                        }
    
    # Create a mapping of column names to their descriptions for each table
    # Use the full descriptions loaded above if available
    column_descriptions = {}
    for table_name, columns in table_to_columns.items():
        column_descriptions[table_name] = {}
        
        if table_name in full_column_descriptions:
            # Use the more comprehensive descriptions
            column_descriptions[table_name] = full_column_descriptions[table_name]
        else:
            # Fallback to the column data from get_column_token_count
            for column in columns:
                col_name = column.get("name", "")
                col_desc = column.get("description", "")
                
                # Store both description and data_type for each column
                column_descriptions[table_name][col_name.lower()] = {
                    "description": col_desc,
                    "data_type": column.get("data_type", ""),
                    "correct_name": col_name  # Keep the correct case
                }
    
    # Prepare the prompt content
    all_tables_columns_info = ""
    for table_name in table_names:
        columns = table_to_columns.get(table_name, [])
        description = table_descriptions.get(table_name, "")
        all_tables_columns_info += f"\nTable: {table_name} - {description}\nColumns:\n"
        
        # Use full descriptions when listing columns in prompt
        for column in columns:
            col_name = column.get("name", "")
            col_desc = ""
            data_type = ""
            
            # Get full description if available
            if table_name in full_column_descriptions and col_name.lower() in full_column_descriptions[table_name]:
                col_info = full_column_descriptions[table_name][col_name.lower()]
                col_desc = col_info.get("description", "")
                data_type = col_info.get("data_type", "")
            else:
                col_desc = column.get("description", "")
                data_type = column.get("data_type", "")
                
            all_tables_columns_info += f"- {col_name} ({data_type}): {col_desc}\n"
    
    # Create the prompt for the LLM
    prompt_template = get_similar_columns_prompt()
    
    formatted_prompt = prompt_template.format(
        all_tables_columns_info=all_tables_columns_info,
        user_query=user_query
    )
    
    # Initialize the LLM
    llm = DEFAULT_COLUMN_FETCHING_LLM
    
    # Calculate prompt token count
    prompt_token_count = get_token_count_for_text(formatted_prompt)
    
    # Get output from LLM - using the correct method for newer LangChain versions
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content=formatted_prompt)]
    response = llm.invoke(messages)
    
    # Extract content from response
    if hasattr(response, "content"):
        response_text = response.content.strip()
    elif isinstance(response, str):
        response_text = response.strip()
    else:
        raise ValueError(f"Unexpected response type: {type(response)}")
    
    # Calculate response token count
    response_token_count = get_token_count_for_text(response_text)
    total_token_count = prompt_token_count + response_token_count
    
    # Parse the response to extract table and column information
    tables_columns_dict = {}
    
    # Parse the pipe-delimited format
    lines = response_text.split('\n')
    table_count = 0
    
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        
        # Split by pipe character
        parts = line.split('|', 1)
        if len(parts) == 2:
            table_name = parts[0].strip()
            column_names = [c.strip() for c in parts[1].split(',') if c.strip()]
            
            # Find the matching table name (case-insensitive)
            actual_table_name = None
            for t_name in table_names:
                if t_name.lower() == table_name.lower():
                    actual_table_name = t_name
                    break
            
            if actual_table_name and column_names:
                table_count += 1
                
                # Find columns information
                columns_list = []
                for col_name in column_names:
                    col_name_lower = col_name.lower()
                    found_description = False
                    
                    # Initialize default values
                    correct_col_name = col_name
                    column_desc = ""  # Initialize with empty string to avoid undefined variables
                    
                    # Look up column description from column_descriptions or full_column_descriptions
                    if actual_table_name in column_descriptions and col_name_lower in column_descriptions[actual_table_name]:
                        col_info = column_descriptions[actual_table_name][col_name_lower]
                        correct_col_name = col_info["correct_name"]
                        column_desc = col_info.get("description", "")
                        found_description = True
                    elif actual_table_name in full_column_descriptions:
                        # Try to find an exact or partial match in the full descriptions
                        for key, info in full_column_descriptions[actual_table_name].items():
                            if col_name_lower == key or col_name_lower in key or key in col_name_lower:
                                correct_col_name = info["correct_name"]
                                column_desc = info.get("description", "")
                                found_description = True
                                break
                    
                    # If still no match, load descriptions directly from the initial_db
                    if not found_description:
                        for db in initial_db:
                            for table in db.get("tables", []):
                                if table.get("name") == actual_table_name:
                                    for column in table.get("columns", []):
                                        col_actual_name = column.get("name", "")
                                        if col_actual_name.lower() == col_name_lower or col_name_lower in col_actual_name.lower():
                                            correct_col_name = col_actual_name
                                            column_desc = column.get("description", "")
                                            found_description = True
                                            break
                                    if found_description:
                                        break
                            if found_description:
                                break
                    
                    # Try one more approach - search all columns case-insensitively
                    if not found_description:
                        for db in initial_db:
                            for table in db.get("tables", []):
                                if table.get("name", "").lower() == actual_table_name.lower():
                                    for column in table.get("columns", []):
                                        if column.get("name", "").lower() == col_name_lower:
                                            column_desc = column.get("description", "")
                                            correct_col_name = column.get("name", col_name)
                                            found_description = True
                                            break
                                    if found_description:
                                        break
                            if found_description:
                                break

                    # If we still don't have a description, generate a meaningful default based on column name
                    if not column_desc:
                        # Generate a description based on the column name
                        words = col_name.replace('_', ' ').split()
                        if col_name_lower.endswith('_id'):
                            entity = ' '.join(words[:-1]) if len(words) > 1 else 'entity'
                            column_desc = f"Unique identifier for the {entity}"
                        elif col_name_lower.startswith('is_') or col_name_lower.startswith('has_'):
                            feature = ' '.join(words[1:])
                            column_desc = f"Boolean flag indicating {feature}"
                        elif 'date' in col_name_lower or 'time' in col_name_lower:
                            column_desc = f"Timestamp for {' '.join(words)}"
                        elif 'price' in col_name_lower or 'cost' in col_name_lower or 'amount' in col_name_lower:
                            column_desc = f"Monetary value for {' '.join(words)}"
                        elif 'count' in col_name_lower or 'num' in col_name_lower or 'qty' in col_name_lower:
                            column_desc = f"Numeric count of {' '.join(words)}"
                        else:
                            column_desc = f"{col_name.replace('_', ' ').title()} information"
                    
                    columns_list.append(ColumnInfo(
                        name=correct_col_name,
                        description=column_desc
                    ))
                
                # Add to tables_columns_dict
                if columns_list:
                    tables_columns_dict[f"table_{table_count}"] = TableColumns(
                        name=actual_table_name,
                        description=table_descriptions.get(actual_table_name, ""),
                        columns=columns_list
                    )
    
    # If no columns were found or parsed, fall back to default approach
    if not tables_columns_dict:
        for idx, table_name in enumerate(table_names):
            # Try to get columns from the full processed database
            if table_name in full_column_descriptions:
                columns_list = []
                for col_name, col_info in list(full_column_descriptions[table_name].items())[:5]:
                    columns_list.append(ColumnInfo(
                        name=col_info["correct_name"],
                        description=col_info["description"]
                    ))
                
                if columns_list:
                    tables_columns_dict[f"table_{idx+1}"] = TableColumns(
                        name=table_name,
                        description=table_descriptions.get(table_name, ""),
                        columns=columns_list
                    )
            else:
                # Fallback to the token count data
                columns = table_to_columns.get(table_name, [])
                
                # Take up to 5 columns per table
                columns_list = []
                for col_info in columns[:5]:
                    col_name = col_info.get("name", "")
                    col_desc = col_info.get("description", "")
                    
                    if col_name:
                        columns_list.append(ColumnInfo(
                            name=col_name,
                            description=col_desc
                        ))
                
                if columns_list:
                    tables_columns_dict[f"table_{idx+1}"] = TableColumns(
                        name=table_name,
                        description=table_descriptions.get(table_name, ""),
                        columns=columns_list
                    )
    
    # Create the result structure
    result = {
        "method": "llm",
        "tokens_used_to_build": total_token_count,
        "tables": tables_columns_dict
    }
    
    return result

# Function to serialize the output to JSON
def get_similar_columns_as_json(user_query: str, table_list: List[Dict[str, Any]], method: str = "vector") -> str:
    """
    Get similar columns and return the result as a JSON string.
    
    Args:
        user_query (str): The user's query about columns
        table_list (List[Dict]): List of tables to search columns in
        method (str): Method to use - "vector" or "llm"
        
    Returns:
        JSON string with the results
    """
    result = get_similar_columns_from_user_query(user_query, table_list, method)
    
    # Convert to JSON with proper handling of Pydantic models
    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, BaseModel):
                return obj.model_dump()  # Use model_dump() instead of dict()
            return super().default(obj)
    
    return json.dumps(result, cls=EnhancedJSONEncoder, indent=2)

# Test the function when run directly
if __name__ == "__main__":
    test_query = input("Enter a query to test column retrieval: ")
    method = input("Enter method (vector/llm): ").lower() or "vector"
    
    try:
        # For testing, use sample table list or get from similar_tables.py
        from similar_tables import get_similar_tables_from_user_query
        
        # Silently get related tables - use suppression to prevent debug output here too
        with io.StringIO() as buf, redirect_stdout(buf):
            tables_result = get_similar_tables_from_user_query(test_query, "vector")[0]
            table_list = [
                {"name": table_info.name, "description": table_info.description} 
                for table_key, table_info in tables_result.get("tables", {}).items()
            ]
        
        if not table_list:
            # Default test table if no tables were found
            table_list = [{"name": "request", "description": "Central table for tickets"}]
        
        # Get results as a JSON string and print it (only output we want to show)
        result_json = get_similar_columns_as_json(test_query, table_list, method)
        print(result_json)
    except Exception as e:
        print(f"Error: {str(e)}")