from langchain.prompts import PromptTemplate
 
# Basic SQL generation prompt
_SQL_GENERATION_TEMPLATE = """"
    # Text-to-SQL Conversion Task
 
    ## Context
    You are a database expert helping to translate natural language queries to SQL for a ticketing system in the 'apolo' schema.
    This is a PostgreSQL database containing tables for request management, users, departments, priorities, etc.
 
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
 
 
_CONTEXT_TEMPLATE = """
# Text-to-SQL Conversion Task
 
role: You are an export Text2SQL generation model. Yout task is to take context and user input and generate SQL query of the user input and return equivalent SQL query.
content: You will be provided with the context of the database and user input. You need to generate SQL query based on the context and user input.
 
# Context
You are a database expert helping to translate natural language queries to SQL for a ticketing system.
 
# Schema Information
"schema_name" "apolo"
"db_description": "This database is designed to store and manage all IT Service Management (ITSM) related data, including service requests, change management records, problem tracking logs, user profiles, asset details, and workflow histories. It serves as the core backend for maintaining the lifecycle of IT operations and support activities.",
 
# Instructions
* ONLY use tables and columns that are explicitly defined in the provided schema.
* ALWAYS use appropriate joins rather than subqueries when possible.
* ALWAYS use case-insensitive string comparisons with LOWER() function if needed.
* DO NOT include any explanations in your response.
* DO NOT make assumptions about tables or columns that are not in the schema. Use only the provided schema.
* ALWAYS handle NULL values appropriately.
* Use onlt the table names and column names that exist in the provided schema. No not change it while sql generation.
* AVOID using deprecated syntax or functions.
* Use schema name "apolo" before each table name
 
# Provided Schema
<schema>
{schema}
</schema>
 
# User Query
<user_query>
{user_query}
</user_query>
 
# Return Format
Return only the following section:
 
## SQL Query
<sql>
Your PostgreSQL query here
</sql>
 
"""
 
 
# New prompt for similar tables retrieval - using consistent naming (with underscore prefix)
_SIMILAR_TABLES_RETRIEVAL_TEMPLATE = """
    You are a database expert who identifies relevant tables based on user queries.
    Available tables:
    {all_tables_info}
    User query: {user_query}
    Your task is to identify ONLY the table names that are most relevant to the user's query.
    DO NOT create new descriptions for tables - I will use the original table descriptions.
    Limit your selection to the 5 most relevant tables.
    Simply identify the table names that would be most helpful for this query.
    FORMAT YOUR RESPONSE LIKE THIS EXACTLY:
    table_name1|This is the description for table 1.
    table_name2|This is the description for table 2.
    Each table should be on a new line with the table name and description separated by a pipe character.
    Do not include any introductory text, explanations, or formatting markers like asterisks or numbers.
"""
 
# New prompt for similar columns retrieval - using consistent naming (with underscore prefix)
_SIMILAR_COLUMNS_RETRIEVAL_TEMPLATE = """
    You are a database expert who identifies relevant columns based on user queries.
    Available tables and columns:
    {all_tables_columns_info}
    User query: {user_query}
    Your task is to identify the columns from these tables that are most relevant to the user's query.
    For each table, select the columns that would be most useful to fulfill the query.
    FORMAT YOUR RESPONSE LIKE THIS EXACTLY:
    table_name|column_name1,column_name2,column_name3
    Each table should be on a new line with the table name and its relevant columns separated by a pipe character.
    The column names should be comma-separated.
    Do not include any introductory text, explanations, or formatting markers.
"""
 
# Create the prompt templates
SQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["db_metadata", "user_query"],
    template=_SQL_GENERATION_TEMPLATE
)
 
 
CONTEXT_SQL_PROMPT = PromptTemplate(
    input_variables=["schema", "user_query"],
    template=_CONTEXT_TEMPLATE
)
 
SIMILAR_TABLES_RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["all_tables_info", "user_query"],
    template=_SIMILAR_TABLES_RETRIEVAL_TEMPLATE
)
 
SIMILAR_COLUMNS_RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["all_tables_columns_info", "user_query"],
    template=_SIMILAR_COLUMNS_RETRIEVAL_TEMPLATE
)