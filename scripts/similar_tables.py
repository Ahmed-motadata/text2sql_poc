import os
import sys
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Add the parent directory to the path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retriever import Retriever
from settings.settings import DEFAULT_TABLE_FETCHING_LLM, get_similar_tables_prompt
from core.helper import get_token_count_for_text, get_table_token_count
from base.vector_store import vector_store

# For structured output parsing
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field as LCField
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Define Pydantic models for structured output
class TableInfo(BaseModel):
    name: str
    description: str

class TablesResult(BaseModel):
    method: str = Field(..., description="Either 'vector' or 'llm'")
    tokens_used_to_build: int = Field(..., description="Token count of user query if vector, total token utilization if llm")
    tables: Dict[str, TableInfo] = Field(..., description="Dictionary of table information")

# LangChain compatible Pydantic models for structured output
class LCTableInfo(LCBaseModel):
    name: str = LCField(description="The name of the table")
    description: str = LCField(description="A description of the table and its relevance to the query")

class LCTablesResult(LCBaseModel):
    tables: Dict[str, LCTableInfo] = LCField(description="Dictionary of table information with table_1, table_2, etc. as keys")

def get_similar_tables_from_user_query(user_query: str, method: str = "vector") -> List[Dict[str, Any]]:
    """
    Retrieve similar tables based on a user query using either vector embeddings or LLM.
    
    Args:
        user_query (str): The user's query about tables
        method (str): Method to use - "vector" or "llm"
        
    Returns:
        List containing a dictionary with the method, token usage, and retrieved tables
    """
    if method.lower() not in ["vector", "llm"]:
        raise ValueError("Method must be either 'vector' or 'llm'")
    
    if method.lower() == "vector":
        result = _get_tables_vector_method(user_query)
    else:  # method == "llm"
        result = _get_tables_llm_method(user_query)
    
    # Convert to list format as requested
    return [result]

def _get_tables_vector_method(user_query: str) -> Dict[str, Any]:
    """
    Use vector embeddings to find similar tables.
    
    Args:
        user_query (str): The user's query
        
    Returns:
        Dict containing method, token count and retrieved tables
    """
    # Calculate token count of user query
    token_counts_user_query = get_token_count_for_text(user_query)
    
    # Set up retriever with appropriate parameters
    k = 10  # Number of results to retrieve
    threshold = 0.7  # Similarity threshold for filtering results
    
    # Create a retriever instance
    retriever = Retriever(k, threshold, user_query)
    
    # Get table documents
    table_docs = retriever.get_table_doc()
    
    # Format the results
    tables_dict = {}
    for idx, (doc, score) in enumerate(table_docs):
        # Get table name from metadata or extract from content
        table_name = doc.metadata.get('table_name')
        if not table_name and "Table:" in doc.page_content:
            table_name = doc.page_content.split('\n')[0].replace('Table:', '').strip()
            
        # If we still don't have a name, use a placeholder
        if not table_name:
            table_name = f"table_{idx+1}"
            
        # Get table description from metadata or content
        description = doc.metadata.get('description', '')
        if not description and len(doc.page_content.split('\n')) > 1:
            # Try to extract description from content
            description_lines = [line for line in doc.page_content.split('\n')[1:] if line.strip()]
            if description_lines:
                description = description_lines[0]
        
        # If no description was found, use a default
        if not description:
            description = f"Table identified with similarity score: {score:.2f}"
            
        # Add to tables dictionary
        tables_dict[f"table_{idx+1}"] = TableInfo(
            name=table_name,
            description=description
        )
    
    # Create the result structure
    result = {
        "method": "vector",
        "tokens_used_to_build": token_counts_user_query,
        "tables": tables_dict
    }
    
    return result

def _get_tables_llm_method(user_query: str) -> Dict[str, Any]:
    """
    Use an LLM to identify similar tables based on the user query.
    
    Args:
        user_query (str): The user's query
        
    Returns:
        Dict containing method, token count and retrieved tables
    """
    # Load database metadata
    from settings.settings import DATABASE_SETTINGS
    input_db_path = DATABASE_SETTINGS["input_db_path"]
    
    try:
        with open(input_db_path, "r") as f:
            initial_db = json.load(f)
    except Exception as e:
        raise Exception(f"Error loading database metadata: {str(e)}")
    
    # Get token counts for all tables
    table_token_counts = get_table_token_count(initial_db, [])
    
    # Create mapping of table names to their original descriptions
    table_descriptions = {}
    for db in initial_db:
        for table in db.get("tables", []):
            table_descriptions[table.get("name")] = table.get("description", "")
    
    # Create a prompt for the LLM using the imported prompt template
    all_tables_info = ""
    for table_info in table_token_counts:
        table_name = table_info.get("table_name", "")
        table_description = table_descriptions.get(table_name, "")
        all_tables_info += f"- {table_name}: {table_description}\n"
    
    # Get the prompt template from settings.py
    prompt_template = get_similar_tables_prompt()
    # Format the prompt with our data
    formatted_prompt = prompt_template.format(
        all_tables_info=all_tables_info,
        user_query=user_query
    )
    
    # Initialize the LLM and get the response - update to new LangChain API
    llm = DEFAULT_TABLE_FETCHING_LLM
    
    # Calculate prompt token count
    prompt_token_count = get_token_count_for_text(formatted_prompt)
    
    # Get structured output from LLM using the new LangChain API
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
    
    # Parse the response to extract table information
    tables_dict = {}
    
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
            
            # Find the matching table name in our schema (case-insensitive)
            actual_table_name = None
            for table_data in table_token_counts:
                schema_table_name = table_data.get("table_name", "")
                if schema_table_name.lower() == table_name.lower():
                    actual_table_name = schema_table_name
                    break
            
            if actual_table_name:
                table_count += 1
                # Use the original description from our database metadata
                tables_dict[f"table_{table_count}"] = TableInfo(
                    name=actual_table_name,
                    description=table_descriptions.get(actual_table_name, "")
                )
    
    # If no tables were parsed from the response, try a different approach
    if not tables_dict:
        # Look for table names in the response
        for table_data in table_token_counts:
            table_name = table_data.get("table_name", "")
            if table_name.lower() in response_text.lower():
                table_count += 1
                tables_dict[f"table_{table_count}"] = TableInfo(
                    name=table_name,
                    description=table_descriptions.get(table_name, "")
                )
                
                # Limit to 5 tables
                if table_count >= 5:
                    break
    
    # If still no tables found, fall back to the most relevant tables by token count
    if not tables_dict:
        # Sort tables by token count as a proxy for relevance
        sorted_tables = sorted(table_token_counts, key=lambda x: x.get("token_count", 0), reverse=True)
        
        for i, table_info in enumerate(sorted_tables[:5]):
            table_name = table_info.get("table_name", "")
            tables_dict[f"table_{i+1}"] = TableInfo(
                name=table_name,
                description=table_descriptions.get(table_name, "")
            )
    
    # Create the result structure
    result = {
        "method": "llm",
        "tokens_used_to_build": total_token_count,
        "tables": tables_dict
    }
    
    return result

# Function to serialize the output to JSON
def get_similar_tables_as_json(user_query: str, method: str = "vector") -> str:
    """
    Get similar tables and return the result as a JSON string.
    
    Args:
        user_query (str): The user's query about tables
        method (str): Method to use - "vector" or "llm"
        
    Returns:
        JSON string with the results
    """
    result = get_similar_tables_from_user_query(user_query, method)
    
    # Convert to JSON with proper handling of Pydantic models
    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, BaseModel):
                return obj.model_dump()  # Use model_dump() instead of dict()
            return super().default(obj)
    
    return json.dumps(result, cls=EnhancedJSONEncoder, indent=2)

# Test the function when run directly
if __name__ == "__main__":
    test_query = input("Enter a query to test table retrieval: ")
    method = input("Enter method (vector/llm): ").lower() or "vector"
    
    try:
        # Get results as a JSON string and print it (structured output only)
        result_json = get_similar_tables_as_json(test_query, method)
        print(result_json)
    except Exception as e:
        print(f"Error: {str(e)}")