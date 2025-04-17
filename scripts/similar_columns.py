import os
import sys
import json
from typing import List

# Add the parent directory to the path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.similar_tables import get_similar_tables_from_user_query

def extract_table_names(similar_tables_output: List[dict]) -> List[str]:
    """
    Extract table names from the similar_tables output.

    Args:
        similar_tables_output (List[dict]): Output from get_similar_tables_from_user_query

    Returns:
        List of table names
    """
    if not similar_tables_output:
        return []

    tables = similar_tables_output[0].get("tables", [])
    return [entry.get("table") for entry in tables if "table" in entry]

if __name__ == "__main__":
    q = input("Enter a query to test table retrieval: ")
    m = input("Enter method (vector/llm): ").lower() or "vector"

    try:
        similar_tables_result = get_similar_tables_from_user_query(q, m)
        table_names = extract_table_names(similar_tables_result)

        print("\nExtracted Table Names:")
        print(json.dumps(table_names, indent=2))

    except Exception as e:
        print(f"Error: {e}")
