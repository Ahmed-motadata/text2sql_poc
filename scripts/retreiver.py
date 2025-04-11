import json
import os
import sys
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Add project root to sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing vector store
from base.vector_store import vector_store

def load_input_json(input_json: Dict[str, str]) -> Dict[str, str]:
    """
    Validates and returns table and columns from the JSON input.
    """
    table = input_json.get("table")
    columns = input_json.get("columns")

    if not table and not columns:
        raise ValueError("Input JSON must contain at least one of 'table' or 'columns'.")

    return {"table": table, "columns": columns}

def retrieve_table_docs(table_name: str) -> List[Document]:
    """
    Retrieves all table-level documents for a given table name.
    Filters based on metadata: type=table and matches table name in content.
    """
    retriever: VectorStoreRetriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    results = retriever.vectorstore.similarity_search_with_score(
        query=f"Table: {table_name}",
        k=1,
        filter={"type": "table"}
    )

    # Filter for exact match on table name in content
    table_docs = [doc for doc, _ in results if f"Table: {table_name}" in doc.page_content]
    return table_docs

def retrieve_column_docs(table_name: str, column_list: List[str]) -> List[Document]:
    """
    Retrieves column-level documents for specific columns of a table.
    Filters based on metadata: type=column and table_name.
    If column_list is empty, return all columns for the table.
    """
    retriever: VectorStoreRetriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 100})
    column_docs = []

    # If no specific columns are passed, fetch all
    if not column_list:
        results = retriever.vectorstore.similarity_search_with_score(
            query=f"All columns of table {table_name}",
            k=100,
            filter={"type": "column", "table_name": table_name}
        )
        column_docs = [doc for doc, _ in results]
    else:
        for column in column_list:
            results = retriever.vectorstore.similarity_search_with_score(
                query=f"Column: {column}",
                k=1,
                filter={"type": "column", "table_name": table_name}
            )
            column_docs.extend([doc for doc, _ in results])

    return column_docs

def run_retrieval_pipeline(input_json: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    """
    Orchestrates table and column retrieval given input JSON.
    Returns dictionary with extracted docs in format:
    {
        "table": [{"request": "table_doc"}],
        "columns": [{"col1": "column_doc"}, {"col2": "column_doc"}]
    }
    """
    parsed = load_input_json(input_json)
    table_name = parsed.get("table")
    columns_str = parsed.get("columns")

    columns = [col.strip() for col in columns_str.split(",")] if columns_str else []

    table_docs = retrieve_table_docs(table_name) if table_name else []
    column_docs = retrieve_column_docs(table_name, columns) if table_name else []

    output = {
        "table": [{table_name: doc.page_content} for doc in table_docs],
        "columns": [{doc.metadata["table_name"] + "." + doc.page_content.split('\n')[0].split(": ")[-1]: doc.page_content} for doc in column_docs]
    }

    return output


# For testing from CLI
if __name__ == "__main__":
    # Example usage:
    input_json = {
        "table": "flotouser",
        "columns": "id,name"
    }

    output = run_retrieval_pipeline(input_json)
    print(json.dumps(output, indent=2))
