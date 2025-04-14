import os
import sys
import json
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Add the parent directory to the path for absolute imports
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
    user_query = "flotouser, uuid, updatedbyid, requests"

    retriever = Retriever(k, threshold, user_query)

    # Retrieve table documents
    table_docs = retriever.get_table_doc()

    print("\nRetrieved Table Documents:")
    if table_docs:
        # Check if there are any table documents
        for doc, score in table_docs:
            print(f"Table Document score: {score:.3f}")
            print(f"Metadata keys: {doc.metadata.keys()}")
            print(f"Table Content: {doc.page_content[:100]}...")  # Display first 100 characters for brevity

        # Extract table names from metadata or content
        table_names = []
        for doc, _ in table_docs:
            table_name = doc.metadata.get('table_name')
            if not table_name and "Table:" in doc.page_content:
                table_name = doc.page_content.split('\n')[0].replace('Table:', '').strip()
            
            if table_name:
                table_names.append(table_name)

        # Deduplicate table names
        table_names = list(set(table_names))

        if not table_names:
            print("Warning: No valid table names were extracted.")
    else:
        print("No table documents found matching the query.")
        table_names = []

    print(f"\nExtracted table names: {table_names}")

    # Retrieve column documents if there are valid table names
    if table_names:
        column_docs = retriever.get_column_doc(table_names)

        print("\nRetrieved Column Documents:")
        column_names = []  # Initialize a list for column names
        if column_docs:
            for doc, score in column_docs:
                table_name = doc.metadata.get('table_name', 'unknown')
                column_name = doc.metadata.get('column_name', 'unknown')
                print(f"[{table_name}.{column_name}] - Score: {score:.3f}")
                print(f"Column Content: {doc.page_content[:100]}...")  # Display first 100 characters for brevity

                # Add column names to the list
                if column_name != 'unknown':
                    column_names.append(f"{table_name}.{column_name}")
        else:
            print("No column documents found matching the query.")
        
        # Deduplicate column names
        column_names = list(set(column_names))
        print(f"\nExtracted column names: {column_names}")
    else:
        print("No table names found, skipping column retrieval.")
