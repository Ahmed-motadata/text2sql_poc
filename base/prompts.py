from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional, Union

# Basic SQL generation prompt
_SQL_GENERATION_TEMPLATE = """"
    # Text-to-SQL Conversion Task

    ## Context
    You are a database expert helping to translate natural language queries to SQL for a ticketing system in the 'apolo' schema.
    This is a PostgreSQL database containing tables for request management, users, departments, priorities, etc.

    ## Database Schema
    {db_metadata}

    ## User Query
    "{user_query}"

    ## Strict Rules
    1. ONLY use tables and columns that are explicitly defined in the schema
    2. ALWAYS include the schema name 'apolo' before each table name (e.g., apolo.request)
    3. ALWAYS use appropriate joins rather than subqueries when possible
    4. ALWAYS use case-insensitive string comparisons with LOWER() function
    5. DO NOT include any explanations in your response
    6. DO NOT make assumptions about tables or columns that are not in the schema
    7. ALWAYS handle NULL values appropriately
    8. ALWAYS add appropriate sorting (typically by creation time DESC)
    9. AVOID using deprecated syntax or functions

    ## Return Format
    Return only the following two sections:

    ### Selected Tables and Columns
    - table_name: column1, column2, column3
    - table_name2: column1, column2
    (List only the tables and columns that are used in your SQL query)

    ### SQL Query
    ```sql
    Your PostgreSQL query here
    ```

    Remember to use only table names and column names that exist in the provided schema.
    """

# SQL generation with examples prompt
_SQL_GENERATION_WITH_EXAMPLES_TEMPLATE = """Given the following database schema:
{schema}

Here are some examples of questions and their corresponding SQL queries:
{examples}

Generate a SQL query to answer the following question:
{question}

SQL Query:"""

# SQL explanation prompt
_SQL_EXPLANATION_TEMPLATE = """Given the following database schema:
{schema}

And the following SQL query:
{query}

Explain what this SQL query is doing in simple terms.
"""

# SQL debugging prompt
_SQL_DEBUGGING_TEMPLATE = """Given the following database schema:
{schema}

The following SQL query has an error:
{query}

Error message:
{error}

Please correct the SQL query and explain the fix.
"""

# SQL optimization prompt
_SQL_OPTIMIZATION_TEMPLATE = """Given the following database schema:
{schema}

The following SQL query works but might not be optimal:
{query}

Please optimize this SQL query for better performance and explain your optimizations.
"""

# Create the prompt templates
SQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["db_metadata", "user_query"],
    template=_SQL_GENERATION_TEMPLATE
)

SQL_GENERATION_WITH_EXAMPLES_PROMPT = PromptTemplate(
    input_variables=["schema", "examples", "question"],
    template=_SQL_GENERATION_WITH_EXAMPLES_TEMPLATE
)

SQL_EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["schema", "query"],
    template=_SQL_EXPLANATION_TEMPLATE
)

SQL_DEBUGGING_PROMPT = PromptTemplate(
    input_variables=["schema", "query", "error"],
    template=_SQL_DEBUGGING_TEMPLATE
)

SQL_OPTIMIZATION_PROMPT = PromptTemplate(
    input_variables=["schema", "query"],
    template=_SQL_OPTIMIZATION_TEMPLATE
)

# Functions to get prompts with customization options
def get_sql_generation_prompt(
    custom_template: Optional[str] = None,
    custom_input_variables: Optional[List[str]] = None
) -> PromptTemplate:
    """
    Returns the SQL generation prompt template.
    
    Args:
        custom_template: Optional custom template to use instead of default
        custom_input_variables: Optional list of input variables for custom template
        
    Returns:
        A PromptTemplate for SQL generation
    """
    if custom_template and custom_input_variables:
        return PromptTemplate(
            input_variables=custom_input_variables,
            template=custom_template
        )
    return SQL_GENERATION_PROMPT

def get_sql_generation_with_examples_prompt(
    custom_template: Optional[str] = None,
    custom_input_variables: Optional[List[str]] = None
) -> PromptTemplate:
    """
    Returns the SQL generation with examples prompt template.
    
    Args:
        custom_template: Optional custom template to use instead of default
        custom_input_variables: Optional list of input variables for custom template
        
    Returns:
        A PromptTemplate for SQL generation with examples
    """
    if custom_template and custom_input_variables:
        return PromptTemplate(
            input_variables=custom_input_variables,
            template=custom_template
        )
    return SQL_GENERATION_WITH_EXAMPLES_PROMPT

def get_sql_explanation_prompt(
    custom_template: Optional[str] = None,
    custom_input_variables: Optional[List[str]] = None
) -> PromptTemplate:
    """
    Returns the SQL explanation prompt template.
    
    Args:
        custom_template: Optional custom template to use instead of default
        custom_input_variables: Optional list of input variables for custom template
        
    Returns:
        A PromptTemplate for SQL explanation
    """
    if custom_template and custom_input_variables:
        return PromptTemplate(
            input_variables=custom_input_variables,
            template=custom_template
        )
    return SQL_EXPLANATION_PROMPT

def get_sql_debugging_prompt(
    custom_template: Optional[str] = None,
    custom_input_variables: Optional[List[str]] = None
) -> PromptTemplate:
    """
    Returns the SQL debugging prompt template.
    
    Args:
        custom_template: Optional custom template to use instead of default
        custom_input_variables: Optional list of input variables for custom template
        
    Returns:
        A PromptTemplate for SQL debugging
    """
    if custom_template and custom_input_variables:
        return PromptTemplate(
            input_variables=custom_input_variables,
            template=custom_template
        )
    return SQL_DEBUGGING_PROMPT

def get_sql_optimization_prompt(
    custom_template: Optional[str] = None,
    custom_input_variables: Optional[List[str]] = None
) -> PromptTemplate:
    """
    Returns the SQL optimization prompt template.
    
    Args:
        custom_template: Optional custom template to use instead of default
        custom_input_variables: Optional list of input variables for custom template
        
    Returns:
        A PromptTemplate for SQL optimization
    """
    if custom_template and custom_input_variables:
        return PromptTemplate(
            input_variables=custom_input_variables,
            template=custom_template
        )
    return SQL_OPTIMIZATION_PROMPT

# Dictionary of all prompts for easy import
ALL_PROMPTS = {
    "generation": SQL_GENERATION_PROMPT,
    "generation_with_examples": SQL_GENERATION_WITH_EXAMPLES_PROMPT,
    "explanation": SQL_EXPLANATION_PROMPT,
    "debugging": SQL_DEBUGGING_PROMPT,
    "optimization": SQL_OPTIMIZATION_PROMPT
}
