import json
import sys
import os
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager, index # Import the index function
# Use SQLRecordManager, assuming standard community integration
# If you installed a specific postgres integration, the path might differ slightly
# from langchain_community.indexes import SQLRecordManager
 
# Add the parent directory to the path to allow imports from base and settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
# Now import the vector_store properly
from base.vector_store import vector_store # Your existing vector_store instance
# Import necessary settings for RecordManager
from settings.settings import PGVECTOR_CONNECTION_STRING, PGVECTOR_COLLECTION_NAME
 
# --- Keep your existing functions, but add 'source' to metadata ---
 
def get_table_docs_from_json(json_path: str):
    """Loads table data from JSON and adds source metadata."""
    with open(json_path, "r") as f:
        data = json.load(f)
 
    table_docs = []
    allowed_metadata_keys = {
        "column_count",
        "table_token_count",
        "table_token_columns_with_columns_name",
        "table_token_with_columns_name_and_description_and_dt"
    }
 
    for table in data["tables"]:
        column_names = [col["name"] for col in table.get("columns", [])]
        table_name = table.get("name", "")
        description = table.get("description", "")
        page_content = f"Table: {table_name}\n\n{description}"
 
        metadata = {}
        for key in allowed_metadata_keys:
            if key in table:
                metadata[key] = table[key]
            # Optional: Keep warning or remove if too verbose
            # else:
            #     print(f"[⚠️ Missing] '{key}' not found in table '{table_name}'")
 
        metadata["column_names"] = column_names
        metadata["type"] = "table"
        # --- Add source metadata ---
        metadata["source"] = json_path # Use the file path as the source identifier
 
        table_docs.append(Document(page_content=page_content.strip(), metadata=metadata))
 
    return table_docs
 
 
def get_column_docs_from_json(json_path: str):
    """Loads column data from JSON and adds source metadata."""
    with open(json_path, "r") as f:
        data = json.load(f)
 
    column_docs = []
 
    for table in data["tables"]:
        for col in table.get("columns", []):
            column_name = col.get("name", "")
            column_description = col.get("description", "")
            page_content = f"Column: {column_name}\n\n{column_description}"
 
            metadata = {}
            for key in [
                "data_type",
                "column_token_count",
                "column_token_count_with_column_name",
                "column_token_count_with_column_name_and_description_and_dt",
            ]:
                if key in col:
                    metadata[key] = col[key]
 
            metadata["table_name"] = table.get("name", "")
            metadata["type"] = "column"
            # --- Add source metadata ---
            metadata["source"] = json_path # Use the file path as the source identifier
 
            column_docs.append(Document(page_content=page_content.strip(), metadata=metadata))
 
    return column_docs
 
 
if __name__ == "__main__":
    try:
        # Use absolute path for better reliability
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        json_path = os.path.join(project_dir, "database", "processed_db.json")
 
        print(f"Loading documents from: {json_path}")
        table_docs = get_table_docs_from_json(json_path)
        column_docs = get_column_docs_from_json(json_path)
 
        if not table_docs and not column_docs:
             print("No documents found in the JSON file. Exiting.")
             sys.exit(0)
 
        if table_docs:
             print("\nSample Table Doc:\n", table_docs[0], "\n")
        
        if column_docs:
             print("Sample Column Doc:\n", column_docs[0], "\n")
 
        all_docs = table_docs + column_docs
        print(f"Loaded {len(all_docs)} total documents (tables + columns).")
 
        # --- Setup Record Manager ---
        # Define a namespace for the RecordManager, often related to the collection
        record_manager_namespace = f"pgvector/{PGVECTOR_COLLECTION_NAME}"
        record_manager = SQLRecordManager(
            record_manager_namespace, db_url=PGVECTOR_CONNECTION_STRING
        )
 
        # --- Create Schema for Record Manager (if it doesn't exist) ---
        try:
             print(f"Creating schema for Record Manager namespace: {record_manager_namespace}")
             record_manager.create_schema()
             print("Schema creation/check completed.")
        except Exception as e:
             print(f"Error during Record Manager schema creation: {e}")
             print("Ensure the database connection string is correct and the server is running.")
             sys.exit(1) # Exit if schema creation fails
 
        # --- Perform Indexed Ingestion ---
        print(f"Starting indexed ingestion with cleanup='full' and source_id_key='source'...")
        try:
            # Use the index function instead of vector_store.add_documents
            indexing_stats = index(
                all_docs,          # Documents to index
                record_manager,    # The RecordManager instance
                vector_store,      # The VectorStore instance
                cleanup="full",    # Cleanup mode ('full' removes docs from vector store not in current batch for this source)
                source_id_key="source", # Metadata key to identify the source of documents
            )
 
            print("\n--- Indexing Stats ---")
            print(f"Documents Added: {indexing_stats.get('num_added', 0)}")
            print(f"Documents Updated: {indexing_stats.get('num_updated', 0)}")
            print(f"Documents Skipped: {indexing_stats.get('num_skipped', 0)}")
            print(f"Documents Deleted: {indexing_stats.get('num_deleted', 0)}")
            print("----------------------")
            print("Indexed ingestion completed successfully.")
 
        except Exception as e:
            print(f"\nError during indexed document ingestion: {e}")
            print("Make sure the PostgreSQL server is running and the credentials are correct.")
            print("Check your .env file and PGVECTOR_CONNECTION_STRING.")
 
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        print("Please check that the processed_db.json file exists in the 'database' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")