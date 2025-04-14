import os
import sys
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Adjust path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.vector_store import vector_store
from settings.settings import get_query_llm


class QueryAnalyzer:
    """
    A class to analyze user queries and retrieve relevant schema information.
    """
    def __init__(self):
        self.llm = get_query_llm()

    def format_schema_context(self, results: Dict[str, Any]) -> str:
        """
        Format the schema information into a context string for SQL generation.

        Args:
            results: The results dictionary from analyze_query

        Returns:
            Formatted schema context as a string
        """
        context = ["### Database Schema Information ###"]

        if results.get("table_docs"):
            context.append("\n## Tables ##")
            for doc, score in results.get("table_docs", []):
                table_info = doc.page_content
                context.append(f"\n# {table_info} (relevance: {score:.2f})")

        column_info = results.get("columns", {})
        if column_info:
            context.append("\n## Columns ##")
            for table_name, columns in column_info.items():
                context.append(f"\n# Table: {table_name}")
                for col in columns:
                    data_type = col.get('data_type', 'unknown')
                    context.append(f"  - {col['name']} ({data_type})")

        return "\n".join(context)

    def calculate_token_and_score_details(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate token counts and scores for tables and columns from the retriever results.

        Args:
            results: The results dictionary containing table and column documents.

        Returns:
            A dictionary with token counts and score details.
        """
        total_table_tokens = 0
        total_column_tokens = 0
        table_details = []
        column_details = []

        # Process table documents
        for doc, score in results.get("table_docs", []):
            table_name = doc.metadata.get("table_name", "unknown")
            token_count = doc.metadata.get("token_count", 0)
            if token_count == 0:
                token_count = doc.metadata.get("table_token_count", 0)
            total_table_tokens += token_count
            table_details.append({
                "table_name": table_name,
                "score": score,
                "token_count": token_count
            })

        # Process column documents
        for doc, score in results.get("column_docs", []):
            table_name = doc.metadata.get("table_name", "unknown")
            column_name = doc.metadata.get("column_name", "unknown")
            
            # Try different token count fields that might be present in the metadata
            token_count = doc.metadata.get("token_count", 0)
            if token_count == 0:
                token_count = doc.metadata.get("column_token_count", 0)
            if token_count == 0:
                # If no specific token count field is found, estimate based on content length
                # This is a fallback approach
                token_count = len(doc.page_content.split()) // 3  # Rough estimation: 3 words ≈ 1 token
                
            total_column_tokens += token_count
            column_details.append({
                "table_name": table_name,
                "column_name": column_name,
                "score": score,
                "token_count": token_count
            })

        total_tokens = total_table_tokens + total_column_tokens

        return {
            "total_tokens": total_tokens,
            "total_table_tokens": total_table_tokens,
            "total_column_tokens": total_column_tokens,
            "table_details": table_details,
            "column_details": column_details
        }

    def display_token_and_score_details(self, details: Dict[str, Any]):
        """
        Display token counts and scores for tables and columns.

        Args:
            details: The dictionary containing token counts and score details.
        """
        print("\nTotal Tokens:", details["total_tokens"])
        print("Total Table Tokens:", details["total_table_tokens"])
        print("Total Column Tokens:", details["total_column_tokens"])

        print("\nTable Details:")
        for table in details["table_details"]:
            print(f"Table: {table['table_name']} | Score: {table['score']:.3f} | Token Count: {table['token_count']}")

        print("\nColumn Details:")
        for column in details["column_details"]:
            print(f"Column: {column['column_name']} | Table: {column['table_name']} | Score: {column['score']:.3f} | Token Count: {column['token_count']}")


class Retriever:
    def __init__(self, k: int, threshold: float, user_query: str):
        self.k = k
        self.threshold = threshold
        self.user_query = user_query

    def get_table_doc(self) -> List[Document]:
        retriever: VectorStoreRetriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
        results = retriever.vectorstore.similarity_search_with_score(
            query=self.user_query,
            k=self.k,
            filter={"type": "table"}
        )

        filtered_docs = []
        for doc, score in results:
            original_score = 1 - score
            if original_score >= self.threshold:
                filtered_docs.append((doc, original_score))

        return filtered_docs

    def get_column_doc(self, table_names: List[str]) -> List[Document]:
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


def generate_sql_from_query(user_query: str, results: Dict[str, Any]) -> str:
    """
    Generate an SQL query based on user input and schema info.
    """
    schema_context = QueryAnalyzer().format_schema_context(results)

    # Gemini expects a single string prompt, not a list of messages.
    full_prompt = (
        "    # Text-to-SQL Conversion Task\n\n    ## Context\n    You are a database expert helping to translate natural language queries to SQL for a ticketing system in the 'apolo' schema.\n    This is a PostgreSQL database containing tables for request management, users, departments, priorities, etc.\n\n    ## Strict Rules\n    1. ONLY use tables and columns that are explicitly defined in the schema\n    2. ALWAYS include the schema name 'apolo' before each table name (e.g., apolo.request)\n    3. ALWAYS use appropriate joins rather than subqueries when possible\n    4. ALWAYS use case-insensitive string comparisons with LOWER() function\n    5. DO NOT include any explanations in your response\n    6. DO NOT make assumptions about tables or columns that are not in the schema\n    7. ALWAYS handle NULL values appropriately\n    8. ALWAYS add appropriate sorting (typically by creation time DESC)\n    9. AVOID using deprecated syntax or functions\n\n    ## Return Format\n    Return only the following two sections:\n\n    ### Selected Tables and Columns\n    - table_name: column1, column2, column3\n    - table_name2: column1, column2\n    (List only the tables and columns that are used in your SQL query)\n\n    ### SQL Query\n    ```sql\n    Your PostgreSQL query here\n    ```\n\n    Remember to use only table names and column names that exist in the provided schema."
        f"{schema_context}\n\n"
        f"User Query: {user_query}"
    )

    llm = get_query_llm()()  # Cached Gemini LLM instance

    try:
        response = llm.invoke(full_prompt)

        # For Gemini: response is usually a `AIMessage` object
        if hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            print(f"Error: Unexpected LLM response type: {type(response)}")
            return ""

    except Exception as e:
        print(f"Error generating SQL: {e}")
        return ""


def format_schema_context(results: Dict[str, Any]) -> str:
    context = ["### Database Schema Information ###"]

    if results.get("table_docs"):
        context.append("\n## Tables ##")
        for doc, score in results.get("table_docs", []):
            table_info = doc.page_content
            context.append(f"\n# {table_info} (relevance: {score:.2f})")

    column_info = results.get("columns", {})
    if column_info:
        context.append("\n## Columns ##")
        for table_name, columns in column_info.items():
            context.append(f"\n# Table: {table_name}")
            for col in columns:
                data_type = col.get('data_type', 'unknown')
                context.append(f"  - {col['name']} ({data_type})")

    return "\n".join(context)


if __name__ == "__main__":
    k = 5
    threshold = 0.7
    user_query = input("Enter your query: ")

    retriever = Retriever(k, threshold, user_query)

    table_docs = retriever.get_table_doc()
    table_names = [doc.metadata.get('table_name') for doc, _ in table_docs if doc.metadata.get('table_name')]
    table_names = list(set(table_names))

    column_docs = retriever.get_column_doc(table_names)

    column_info = {}
    for doc, score in column_docs:
        table_name = doc.metadata.get('table_name', 'unknown')
        column_name = doc.metadata.get('column_name', 'unknown')

        if table_name not in column_info:
            column_info[table_name] = []

        column_info[table_name].append({
            "name": column_name,
            "data_type": doc.metadata.get('data_type', 'unknown'),
            "score": score,
            "content": doc.page_content
        })

    results = {
        "query": user_query,
        "tables": table_names,
        "columns": column_info,
        "table_docs": table_docs,
        "column_docs": column_docs
    }

    # Calculate and display token and score details
    analyzer = QueryAnalyzer()
    details = analyzer.calculate_token_and_score_details(results)
    analyzer.display_token_and_score_details(details)

    sql_query = generate_sql_from_query(user_query, results)

    print("\n==================================================")
    print("Generated SQL Query:")
    print("==================================================")
    print(sql_query)
    print("==================================================")
