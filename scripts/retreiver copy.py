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
    Modified to match the enhanced column name emphasis in the ingestion process.
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
            # Create a query that matches our enhanced column content structure
            # This emphasizes the column name in the same way it was emphasized during ingestion
            query = f"Column: {column}\nCOLUMN_NAME: {column}\nCOLUMN_ID: {column}\nTABLE: {table_name}"
            
            # Get top 5 results for potential matches
            results = retriever.vectorstore.similarity_search_with_score(
                query=query,
                k=5,
                filter={"type": "column", "table_name": table_name}
            )
            
            if not results:
                continue
                
            # Try to find an exact match first
            exact_matches = []
            for doc, score in results:
                # Extract column name from the content (first line after "Column: ")
                content_lines = doc.page_content.split('\n')
                if content_lines and content_lines[0].startswith("Column:"):
                    doc_column_name = content_lines[0].replace("Column:", "").strip()
                    
                    # Check for exact match (case insensitive)
                    if doc_column_name.lower() == column.lower():
                        exact_matches.append((doc, score))
            
            # If exact matches found, use the first one
            if exact_matches:
                column_docs.append(exact_matches[0][0])
            # Otherwise use the top result from semantic search
            elif results:
                column_docs.append(results[0][0])
 
    return column_docs
 
def run_retrieval_pipeline(input_json_list: List[Dict[str, str]]) -> List[Dict[str, any]]:
    """
    Accepts a list of input JSONs for multiple tables and columns.
    Returns a list where each table is followed by its columns.
    """
    output_list = []
 
    for input_json in input_json_list:
        parsed = load_input_json(input_json)
        table_name = parsed.get("table")
        columns_str = parsed.get("columns")
 
        columns = [col.strip() for col in columns_str.split(",")] if columns_str else []
 
        # Debug output for troubleshooting
        print(f"Searching for table: {table_name}")
        print(f"Searching for columns: {columns}")
 
        table_docs = retrieve_table_docs(table_name) if table_name else []
        column_docs = retrieve_column_docs(table_name, columns) if table_name else []
 
        # Debug the retrieved column documents
        print(f"Retrieved {len(column_docs)} column documents")
        for i, doc in enumerate(column_docs):
            # Extract column name from the first line
            col_name = doc.page_content.split('\n')[0].replace("Column:", "").strip()
            print(f"Column {i+1}: {col_name} (matching input: {columns[i] if i < len(columns) else 'unknown'})")
 
        # Add table document
        if table_docs:
            output_list.append({"table": {table_name: table_docs[0].page_content}})
            
            # Add column documents for this table
            if column_docs:
                output_list.append({
                    "columns": [
                        {doc.metadata["table_name"] + "." + doc.page_content.split('\n')[0].split(": ")[-1]: 
                         # Clean up the output to show only the column and description
                         "\n".join([doc.page_content.split('\n')[0], "", doc.page_content.split('\n\n')[-1]])}
                        for doc in column_docs
                    ]
                })
 
    return output_list
 
if __name__ == "__main__":
    # Example usage with multiple tables:
    input_json_list = [
        {
            "table": "priority",
            "columns": "id,name"
        },
        {
            "table": "request",
            "columns": "I want to see who has requested ticket no 101"
        },
        
    ]

    # If you want to use just a single table, uncomment this:
    # input_json = {
    #     "table": "flotouser",
    #     "columns": "id,name"
    # }
    # input_json_list = [input_json]  # Convert to list format

    output = run_retrieval_pipeline(input_json_list)
    print(json.dumps(output, indent=2))
