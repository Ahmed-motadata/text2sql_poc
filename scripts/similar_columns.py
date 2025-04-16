import os
import sys
import json
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.retriever import Retriever
from settings.settings import DEFAULT_COLUMN_FETCHING_LLM, get_similar_columns_prompt
from core.helper import get_token_count_for_text, get_table_token_count, get_column_token_count
from base.vector_store import vector_store


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
    import logging
    from langchain_community.callbacks import get_openai_callback
    from settings.settings import DATABASE_SETTINGS

    logging.basicConfig(level=logging.INFO)
    
    # Load database metadata
    input_db_path = DATABASE_SETTINGS["input_db_path"]
    processed_db_path = DATABASE_SETTINGS["output_path"]["processed_db"]
    
    try:
        with open(processed_db_path, "r") as f:
            processed_db = json.load(f)
            initial_db = [processed_db]
            logging.info(f"Loaded processed database from {processed_db_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Failed to load processed database: {e}")
        try:
            with open(input_db_path, "r") as f:
                initial_db = json.load(f)
                logging.info(f"Loaded raw database from {input_db_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading database metadata: {e}")
            raise
    
    # Extract table names
    table_names = [table['name'] for table in table_list if 'name' in table]
    if not table_names:
        logging.error("No valid table names provided in table_list")
        raise ValueError("No valid table names provided in table_list")
    
    # Get column information
    column_input = [{"table": table_name, "column": None} for table_name in table_names]
    columns_data = get_column_token_count(initial_db, column_input)
    
    table_to_columns = {}
    for table_info in columns_data:
        table_name = table_info.get("table_name", "")
        if not table_name:
            logging.warning(f"Skipping table info with missing table_name: {table_info}")
            continue
        table_to_columns[table_name] = table_info.get("columns", [])
    
    # Load column descriptions
    full_column_descriptions = {}
    for db in initial_db:
        for table in db.get("tables", []):
            table_name = table.get("name", "")
            if table_name:
                if table_name in full_column_descriptions:
                    logging.warning(f"Duplicate table name found: {table_name}")
                full_column_descriptions[table_name] = {}
                for column in table.get("columns", []):
                    col_name = column.get("name", "")
                    if col_name:
                        full_column_descriptions[table_name][col_name.lower()] = {
                            "description": column.get("description", ""),
                            "data_type": column.get("data_type", ""),
                            "correct_name": col_name
                        }
    
    # Prepare prompt
    table_descriptions = {table['name']: table.get('description', '') for table in table_list}
    all_tables_columns_info = ""
    for table_name in table_names:
        columns = table_to_columns.get(table_name, [])
        all_tables_columns_info += f"\nTable: {table_name}\nColumns:\n"
        for column in columns:
            col_name = column.get("name", "")
            col_desc = ""
            data_type = ""
            if table_name in full_column_descriptions and col_name.lower() in full_column_descriptions[table_name]:
                col_info = full_column_descriptions[table_name][col_name.lower()]
                col_desc = col_info.get("description", "")
                data_type = col_info.get("data_type", "")
            else:
                col_desc = column.get("description", "")
                data_type = column.get("data_type", "")
            all_tables_columns_info += f"- {col_name} ({data_type}): {col_desc}\n"
    
    prompt_template = get_similar_columns_prompt()
    formatted_prompt = prompt_template.format(
        all_tables_columns_info=all_tables_columns_info,
        user_query=user_query
    )
    logging.debug(f"Formatted prompt:\n{formatted_prompt}")
    
    # Invoke LLM
    llm = DEFAULT_COLUMN_FETCHING_LLM
    try:
        with get_openai_callback() as cb:
            response = llm.invoke(formatted_prompt)
            prompt_tokens = cb.prompt_tokens
            total_tokens = cb.total_tokens
            response_text = response.content if hasattr(response, "content") else str(response)
        #logging.info(f"Used LangChain callback tracking: {total_tokens} tokens")
    except Exception as e:
        logging.error(f"LLM invocation failed: {e}")
        return {"method": "llm", "tokens_used_to_build": -1, "tables": {}}
    
    # Parse response
    tables_columns_dict = {}
    response_text = response_text.strip()
    lines = response_text.split('\n')
    table_count = 0
    
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split('|', 1)
        if len(parts) != 2:
            logging.warning(f"Invalid response line: {line}")
            continue
        table_name = parts[0].strip()
        column_names = [c.strip() for c in parts[1].split(',') if c.strip()]
        actual_table_name = None
        for t_name in table_names:
            if t_name.lower() == table_name.lower():
                actual_table_name = t_name
                break
        if actual_table_name and column_names:
            table_count += 1
            columns_list = []
            for col_name in column_names:
                col_name_lower = col_name.lower()
                correct_col_name = col_name
                column_desc = f"Description not available for {col_name}"
                if actual_table_name in full_column_descriptions and col_name_lower in full_column_descriptions[actual_table_name]:
                    col_info = full_column_descriptions[actual_table_name][col_name_lower]
                    correct_col_name = col_info["correct_name"]
                    column_desc = col_info.get("description", column_desc)
                columns_list.append(ColumnInfo(
                    name=correct_col_name,
                    description=column_desc
                ))
            if columns_list:
                tables_columns_dict[f"table_{table_count}"] = TableColumns(
                    name=actual_table_name,
                    description=table_descriptions.get(actual_table_name, ""),
                    columns=columns_list
                )
    
    # Fallback
    if not tables_columns_dict:
        logging.warning("No columns found in LLM response; using fallback")
        for idx, table_name in enumerate(table_names):
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
    
    return {
        "method": "llm",
        #"prompt_tokens" : prompt_tokens,
        "tokens_used_to_build": total_tokens,
        "tables": tables_columns_dict
    }

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


if __name__ == "__main__":
    test_query = input("Enter a query to test column retrieval: ")
    method = input("Enter method (vector/llm): ").lower() or "vector"
    
    try:
        # Static table list
        table_list = [
            {"name": "request", "description": ""},
            {"name": "flotouser", "description": ""},
            {"name": "status", "description": ""},
            {"name": "priority", "description": ""},
            {"name": "impact", "description": ""},
            {"name": "department", "description": ""},
            {"name": "location", "description": ""},
            {"name": "product", "description": ""},
            {"name": "change", "description": ""},
            {"name": "release", "description": ""},
            {"name": "task", "description": ""},
            {"name": "category", "description": ""},
            {"name": "company", "description": ""},
            {"name": "vendor", "description": ""},
            {"name": "problem", "description": ""},
            {"name": "project", "description": ""},
            {"name": "reportcolumn", "description": ""},
            {"name": "reportdefinition", "description": ""},
            {"name": "patch", "description": ""},
            {"name": "flotorole", "description": ""}
        ]
        
        # Get results as a JSON string and print it
        result_json = get_similar_columns_as_json(test_query, table_list, method)
        print(result_json)
    except Exception as e:
        print(f"Error: {str(e)}")