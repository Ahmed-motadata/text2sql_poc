import psycopg2
from psycopg2.extras import execute_values
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

def connect_to_pgvector(
    dbname=None,
    user=None,
    password=None,
    host="localhost",
    port=5432,
    timeout=None,
    **kwargs
):
    """
    Establishes a connection to the PostgreSQL database with pgvector extension.
    
    Args:
        dbname (str): Name of the database to connect to.
        user (str): Username for authentication.
        password (str): Password for authentication.
        host (str): Hostname or IP address of the PostgreSQL server. Default is localhost.
        port (int): Port of the PostgreSQL server. Default is 5432.
        timeout (float, optional): Timeout for connection in seconds.
        **kwargs: Additional arguments to pass to the psycopg2.connect function.
        
    Returns:
        tuple: (connection, cursor) - Connection and cursor objects for the PostgreSQL database.
        
    Raises:
        Exception: Any error that occurs during connection attempt.
    """
    try:
        logging.info(f"Connecting to PostgreSQL with pgvector at {host}:{port}")
        connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            connect_timeout=timeout,
            **kwargs
        )
        # Create a cursor
        cursor = connection.cursor()
        
        # Check if pgvector extension is installed
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if cursor.fetchone() is None:
            logging.warning("pgvector extension not found. Attempting to install...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.commit()
            logging.info("pgvector extension installed successfully")
        
        logging.info("Successfully connected to PostgreSQL with pgvector")
        return connection, cursor
    except Exception as e:
        logging.error(f"Failed to connect to PostgreSQL with pgvector: {e}")
        raise

def create_vector_table(
    connection,
    cursor,
    table_name="schema_embeddings",
    schema_name="public",
    vector_size=768
):
    """
    Creates a table for storing vector embeddings if it doesn't exist.
    
    Args:
        connection: PostgreSQL connection object
        cursor: PostgreSQL cursor object
        table_name (str): Name of the table to create
        schema_name (str): Name of the schema to create the table in
        vector_size (int): Dimension of the vector embeddings
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if the table already exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            )
        """, (schema_name, table_name))
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logging.info(f"Creating vector table {schema_name}.{table_name}")
            # Create the table for storing vector embeddings
            cursor.execute(f"""
                CREATE TABLE {schema_name}.{table_name} (
                    id SERIAL PRIMARY KEY,
                    item_id TEXT NOT NULL UNIQUE,
                    item_type TEXT NOT NULL,
                    schema_name TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    description TEXT,
                    metadata JSONB,
                    embedding vector({vector_size})
                );
            """)
            
            # Create index for item_type and schema_name for filtering
            cursor.execute(f"""
                CREATE INDEX ON {schema_name}.{table_name} (item_type);
            """)
            cursor.execute(f"""
                CREATE INDEX ON {schema_name}.{table_name} (schema_name);
            """)
            
            # Create vector index for similarity search
            cursor.execute(f"""
                CREATE INDEX ON {schema_name}.{table_name} 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            # Commit the changes
            connection.commit()
            logging.info(f"Successfully created table {schema_name}.{table_name}")
        else:
            logging.info(f"Table {schema_name}.{table_name} already exists")
        
        return True
    except Exception as e:
        connection.rollback()
        logging.error(f"Failed to create vector table: {e}")
        return False

def insert_vectors(
    connection,
    cursor,
    vectors: List[Dict[str, Any]],
    table_name="schema_embeddings",
    schema_name="public",
    batch_size=100
):
    """
    Insert vectors into the pgvector table.
    
    Args:
        connection: PostgreSQL connection object
        cursor: PostgreSQL cursor object
        vectors: List of dictionaries containing:
            - item_id (str): Unique identifier
            - item_type (str): Type of item (table, column, etc.)
            - schema_name (str): Database schema name
            - item_name (str): Name of the item
            - description (str, optional): Description of the item
            - metadata (dict, optional): Additional metadata
            - embedding (list): Vector embedding
        table_name (str): Name of the table to insert into
        schema_name (str): Schema of the table
        batch_size (int): Number of vectors to insert in each batch
        
    Returns:
        int: Number of vectors inserted
    """
    if not vectors:
        logging.warning("No vectors provided for insertion")
        return 0
    
    try:
        # Prepare data for batch insertion
        inserted_count = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            values = []
            
            for vector in batch:
                values.append((
                    vector['item_id'],
                    vector['item_type'],
                    vector['schema_name'],
                    vector['item_name'],
                    vector.get('description', ''),
                    vector.get('metadata', {}),
                    vector['embedding']
                ))
            
            # Execute batch insert
            execute_values(
                cursor,
                f"""
                INSERT INTO {schema_name}.{table_name}
                (item_id, item_type, schema_name, item_name, description, metadata, embedding)
                VALUES %s
                ON CONFLICT (item_id) DO UPDATE SET
                    item_type = EXCLUDED.item_type,
                    schema_name = EXCLUDED.schema_name,
                    item_name = EXCLUDED.item_name,
                    description = EXCLUDED.description,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                values
            )
            
            connection.commit()
            inserted_count += len(batch)
            logging.info(f"Inserted/updated batch of {len(batch)} vectors")
        
        return inserted_count
    except Exception as e:
        connection.rollback()
        logging.error(f"Failed to insert vectors: {e}")
        raise

def search_similar_items(
    cursor,
    query_vector: List[float],
    table_name="schema_embeddings",
    schema_name="public",
    filters: Dict[str, Any] = None,
    limit=10
) -> List[Dict[str, Any]]:
    """
    Search for items similar to the query vector.
    
    Args:
        cursor: PostgreSQL cursor object
        query_vector: Vector embedding to search with
        table_name: Name of the table to search in
        schema_name: Schema name of the table
        filters: Dictionary of filters to apply:
            - item_type: Type of items to return (table, column, etc.)
            - schema_name: Database schema to filter by
        limit: Maximum number of results to return
        
    Returns:
        List of dictionaries with search results including similarity scores
    """
    try:
        # Build the WHERE clause based on filters
        where_clauses = []
        params = [query_vector]
        
        if filters:
            if 'item_type' in filters:
                where_clauses.append("item_type = %s")
                params.append(filters['item_type'])
            
            if 'schema_name' in filters:
                where_clauses.append("schema_name = %s")
                params.append(filters['schema_name'])
        
        # Construct the final WHERE clause
        where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"
        
        # Execute the query with cosine similarity
        cursor.execute(f"""
            SELECT 
                item_id,
                item_type,
                schema_name,
                item_name,
                description,
                metadata,
                1 - (embedding <=> %s) AS similarity
            FROM {schema_name}.{table_name}
            WHERE {where_clause}
            ORDER BY similarity DESC
            LIMIT %s
        """, params + [limit])
        
        # Process results
        results = []
        for row in cursor.fetchall():
            results.append({
                'item_id': row[0],
                'item_type': row[1],
                'schema_name': row[2],
                'item_name': row[3],
                'description': row[4],
                'metadata': row[5],
                'similarity': row[6]
            })
        
        return results
    except Exception as e:
        logging.error(f"Failed to search for similar items: {e}")
        raise

def close_connection(connection, cursor):
    """
    Close the database connection and cursor.
    
    Args:
        connection: PostgreSQL connection object
        cursor: PostgreSQL cursor object
    """
    if cursor:
        cursor.close()
    if connection:
        connection.close()
        logging.info("PostgreSQL connection closed")

def get_relevant_schema_for_query(
    cursor,
    query_embedding: List[float],
    table_name="schema_embeddings",
    schema_name="public",
    db_schema_filter: str = None,
    table_limit=5,
    column_limit_per_table=10
) -> Dict[str, Any]:
    """
    Get relevant schema objects (tables and columns) for a natural language query.
    
    Args:
        cursor: PostgreSQL cursor object
        query_embedding: Vector embedding of the query
        table_name: Name of the vector table
        schema_name: Schema name of the vector table
        db_schema_filter: Optional database schema to filter by (e.g., 'public', 'apolo')
        table_limit: Maximum number of tables to return
        column_limit_per_table: Maximum number of columns to return per table
        
    Returns:
        Dictionary with relevant tables and their columns
    """
    # Get relevant tables
    table_filters = {'item_type': 'table'}
    if db_schema_filter:
        table_filters['schema_name'] = db_schema_filter
    
    tables = search_similar_items(
        cursor,
        query_embedding,
        table_name,
        schema_name,
        filters=table_filters,
        limit=table_limit
    )
    
    result = {
        'tables': []
    }
    
    # For each table, get its relevant columns
    for table in tables:
        table_info = {
            'schema': table['schema_name'],
            'name': table['item_name'],
            'description': table['description'],
            'similarity': table['similarity'],
            'columns': []
        }
        
        # Get columns for this table
        cursor.execute(f"""
            SELECT 
                item_id,
                item_type,
                schema_name,
                item_name,
                description,
                metadata,
                1 - (embedding <=> %s) AS similarity
            FROM {schema_name}.{table_name}
            WHERE item_type = 'column' 
            AND schema_name = %s 
            AND metadata->>'table' = %s
            ORDER BY similarity DESC
            LIMIT %s
        """, [query_embedding, table['schema_name'], table['item_name'], column_limit_per_table])
        
        for row in cursor.fetchall():
            column_info = {
                'name': row[3],
                'description': row[4],
                'data_type': row[5].get('data_type') if row[5] else None,
                'similarity': row[6]
            }
            table_info['columns'].append(column_info)
        
        result['tables'].append(table_info)
    
    return result



    """
    
    
from langchain_postgres import PGVector
from embed_models import get_jina_embed_base
 
connection = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"
collection_name = "T2sql"
 
vector_store = PGVector(
    embeddings=get_jina_embed_base(),
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
    """