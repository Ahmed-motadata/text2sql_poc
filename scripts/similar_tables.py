import os
import sys
import json
from typing import List, Dict, Any

# Add the parent directory to the path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retriever import Retriever
from settings.settings import DEFAULT_TABLE_FETCHING_LLM, get_similar_tables_prompt, DATABASE_SETTINGS
from core.helper import get_token_count_for_text, get_table_token_count

def get_similar_tables_from_user_query(user_query: str, method: str = "vector") -> List[Dict[str, Any]]:
    """
    Retrieve similar tables based on a user query using either vector embeddings or LLM.

    Args:
        user_query (str): The user's query about tables
        method (str): Method to use - "vector" or "llm"

    Returns:
        List containing a dictionary with the method and retrieved tables
    """
    if method.lower() not in ["vector", "llm"]:
        raise ValueError("Method must be either 'vector' or 'llm'")

    if method.lower() == "vector":
        result = _get_tables_vector_method(user_query)
    else:
        result = _get_tables_llm_method(user_query)

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

    llm = DEFAULT_TABLE_FETCHING_LLM
    raw_response = llm.invoke(prompt)

    content = getattr(raw_response, 'content', str(raw_response))

    prettified_tables = []
    for idx, line in enumerate(content.strip().split("\n"), start=1):
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                prettified_tables.append({"table": parts[0].strip(), "description": parts[1].strip()})

    input_tokens = None
    output_tokens = None
    if hasattr(raw_response, 'response_metadata'):
        token_data = raw_response.response_metadata.get('token_usage', {})
        input_tokens = token_data.get('prompt_tokens')
        output_tokens = token_data.get('completion_tokens')

    return {
        "method": "llm",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tables": prettified_tables
    }

def get_similar_tables_as_json(user_query: str, method: str = "vector") -> str:
    """
    Get similar tables and return the result as a JSON string.
    """
    result = get_similar_tables_from_user_query(user_query, method)[0]
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    q = input("Enter a query to test table retrieval: ")
    m = input("Enter method (vector/llm): ").lower() or "vector"
    try:
        out = get_similar_tables_from_user_query(q, m)[0]
        print(json.dumps({"tables": out.get("tables", [])}, indent=2))
    except Exception as e:
        print(f"Error: {e}")
