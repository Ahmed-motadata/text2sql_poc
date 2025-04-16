import sys
import os
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
 
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from base.vector_store import vector_store
 
 
class Retriever:
    """
    A retriever class to obtain relevant table and column documents
    from a vector store based on a user query.
    
    Attributes:
        k (int): Number of results to retrieve.
        threshold (float): Similarity threshold for retrieval.
        user_query (str): User query for retrieval.
    """
 
    def __init__(self, k: int, threshold: float, user_query: str):
        """
        Initialize the Retriever.
 
        Args:
            k (int): Number of results to retrieve.
            threshold (float): Similarity threshold for filtering results.
            user_query (str): User query for retrieval.
        """
        self.k = k
        self.threshold = threshold
        self.user_query = user_query
 
    def get_table_doc(self) -> List[Document]:
        """
        Retrieve table-level documents relevant to the user query,
        with deduplication based on the table name.
 
        Returns:
            List[Document]: A list of unique table documents.
        """
        retriever: VectorStoreRetriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
        results = retriever.vectorstore.similarity_search_with_score(
            query=self.user_query,
            k=self.k,
            filter={"type": "table"}
        )
 
        filtered_docs_with_threshold = []
        for doc, score in results:
            original_score = 1 - score
            if original_score >= self.threshold:
                filtered_docs_with_threshold.append((doc, original_score))
 
        return filtered_docs_with_threshold
 
    def get_column_doc(self, table_names: List[str]) -> List[Document]:
        """
        Retrieve column-level documents for the given list of tables based on the user query,
        with deduplication based on a table.column key.
 
        Args:
            table_names (List[str]): List of table names to retrieve columns for.
 
        Returns:
            List[Document]: A list of unique column documents.
        """
        retriever: VectorStoreRetriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
 
        final_result = []
 
        for table_name in table_names:
            results = retriever.vectorstore.similarity_search_with_score(
                query=self.user_query,
                k=self.k,
                filter={"type": "column", "table_name": table_name}
            )
 
            for doc, score in results:
                original_score = 1 - score
                if original_score >= self.threshold:
                    final_result.append((doc, original_score))
 
        return final_result
 
 
if __name__ == "__main__":
    k = 5
    threshold = 0.7
    user_query = "Show me 5 requests with high priority"
 
    retriever = Retriever(k, threshold, user_query)
 
    # Retrieve table documents
    table_docs = retriever.get_table_doc()
    table_names = list({doc.metadata["table_name"] for doc, _ in table_docs})
    print("\nRetrieved Table Documents:")
    for doc in table_docs:
       print(doc)
        
 
    # Retrieve column documents
    column_docs = retriever.get_column_doc(table_names)
    
 
    print("\nRetrieved Column Documents:")
    for doc, score in column_docs:
        print(f"[{doc.metadata['table_name']}.{doc.metadata['column_name']}] - Score: {score:.3f}")
        print(doc)
 