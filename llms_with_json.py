import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

# Adjust path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings.settings import get_query_llm


class QueryAnalyzer:
    """
    A class to analyze user queries and retrieve relevant schema information.
    Uses processed JSON data instead of vector embeddings.
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
            for table_info, score in results.get("table_docs", []):
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
        Calculate token counts and scores for tables and columns from the results.

        Args:
            results: The results dictionary containing table and column information.

        Returns:
            A dictionary with token counts and score details.
        """
        total_table_tokens = 0
        total_column_tokens = 0
        table_details = []
        column_details = []

        # Process table documents
        for table_info, score in results.get("table_docs", []):
            # Get the table name
            table_name = table_info.get("name", "unknown")
            
            # Get the token count from the table_token_count field
            token_count = table_info.get("table_token_count", 0)
            
            total_table_tokens += token_count
            table_details.append({
                "table_name": table_name,
                "score": score,
                "token_count": token_count
            })

        # Process column documents
        for column_info, score in results.get("column_docs", []):
            # Get table and column names
            table_name = column_info.get("table_name", "unknown")
            column_name = column_info.get("name", "unknown")
            
            # Get the token count from the column_token_count_with_column_name_and_description_and_dt field
            token_count = column_info.get("column_token_count_with_column_name_and_description_and_dt", 0)
            
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


class JSONRetriever:
    """
    A class to retrieve relevant database schema information from the processed JSON data.
    """
    def __init__(self, k: int, threshold: float, user_query: str):
        self.k = k
        self.threshold = threshold
        self.user_query = user_query.lower()
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "processed_db.json")
        self.db_data = self._load_db_data()

    def _load_db_data(self) -> Dict[str, Any]:
        """Load the processed JSON database."""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
            return {"tables": []}

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Clean and normalize text
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Calculate similarity
        return SequenceMatcher(None, text1, text2).ratio()

    def get_table_doc(self) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get relevant tables based on the user query.
        
        Returns:
            List of (table_info, score) tuples
        """
        results = []
        
        for table in self.db_data.get("tables", []):
            # Calculate similarity based on table name and description
            name_sim = self._calculate_similarity(self.user_query, table.get("name", ""))
            desc_sim = self._calculate_similarity(self.user_query, table.get("description", ""))
            
            # Use the higher similarity score, with a bias towards name matches
            score = max(name_sim * 1.2, desc_sim)
            
            if score >= self.threshold:
                results.append((table, score))
        
        # Sort by score and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.k]

    def get_column_doc(self, table_names: List[str]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get relevant columns for the specified tables based on the user query.
        
        Args:
            table_names: List of table names to search columns for
            
        Returns:
            List of (column_info, score) tuples
        """
        results = []
        
        for table in self.db_data.get("tables", []):
            if table.get("name") in table_names:
                table_name = table.get("name")
                
                for column in table.get("columns", []):
                    # Add table name reference to the column
                    column["table_name"] = table_name
                    
                    # Calculate similarity based on column name and description
                    name_sim = self._calculate_similarity(self.user_query, column.get("name", ""))
                    desc_sim = self._calculate_similarity(self.user_query, column.get("description", ""))
                    
                    # Use the higher similarity score, with a bias towards name matches
                    score = max(name_sim * 1.2, desc_sim)
                    
                    if score >= self.threshold:
                        results.append((column, score))
        
        # Sort by score and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.k]


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


if __name__ == "__main__":
    k = 5
    threshold = 0.3  # Lower threshold for text similarity compared to vector similarity
    user_query = input("Enter your query: ")

    retriever = JSONRetriever(k, threshold, user_query)

    table_docs = retriever.get_table_doc()
    table_names = [doc.get('name') for doc, _ in table_docs if doc.get('name')]
    
    column_docs = retriever.get_column_doc(table_names)

    # Format table_docs for display
    formatted_table_docs = []
    for table, score in table_docs:
        table_info = f"Table: {table.get('name')} - {table.get('description', '')[:100]}..."
        formatted_table_docs.append((table_info, score))

    # Organize columns by table
    column_info = {}
    for doc, score in column_docs:
        table_name = doc.get('table_name', 'unknown')
        
        if table_name not in column_info:
            column_info[table_name] = []

        column_info[table_name].append({
            "name": doc.get('name', 'unknown'),
            "data_type": doc.get('data_type', 'unknown'),
            "score": score,
            "content": doc.get('description', '')
        })

    results = {
        "query": user_query,
        "tables": table_names,
        "columns": column_info,
        "table_docs": table_docs,
        "column_docs": column_docs
    }

    # Prepare for schema context formatting
    results_for_context = {
        "query": user_query,
        "tables": table_names,
        "columns": column_info,
        "table_docs": formatted_table_docs
    }

    # Calculate and display token and score details
    analyzer = QueryAnalyzer()
    details = analyzer.calculate_token_and_score_details(results)
    analyzer.display_token_and_score_details(details)

    sql_query = generate_sql_from_query(user_query, results_for_context)

    print("\n==================================================")
    print("Generated SQL Query:")
    print("==================================================")
    print(sql_query)
    print("==================================================")