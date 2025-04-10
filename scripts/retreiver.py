import sys
import os
 
# Add the parent directory to the path to allow imports from base
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
# Import the vector_store module correctly
from base.vector_store import vector_store

def search_similar_schema(query, k=1):
    """
    Search for schema elements similar to the given query.
    
    Args:
        query: The search query
        k: Number of results to return
        
    Returns:
        List of matching documents
    """
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        print(f"Error searching vector store: {e}")
        print("Make sure the PostgreSQL server is running and the credentials are correct.")
        return []
 
if __name__ == "__main__":
    query = "Show me request table's columns?"
    results = search_similar_schema(query)
    
    for doc in results:
        print(doc.page_content)
        print(doc.metadata)