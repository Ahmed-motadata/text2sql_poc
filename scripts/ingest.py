import json
import sys
import os
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager, index
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.vector_store import vector_store
from settings.settings import PGVECTOR_CONNECTION_STRING, PGVECTOR_COLLECTION_NAME
from core.helper import run_get_token_count

def generate_processed_db(input_db_path=None):
    """Generate the processed_db.json file using the helper function."""
    try:
        result = run_get_token_count(input_db_path)
        return True
    except Exception as e:
        print(f"Error generating processed_db.json: {e}")
        return False

def clear_vector_store():
    """Completely clears the vector store of all documents."""
    try:
        if hasattr(vector_store, "delete"):
            vector_store.delete(ids=None, filter=None)
            print("Vector store has been completely cleared.")
        else:
            print("WARNING: Could not clear vector store automatically. Manual deletion may be needed.")
    except Exception as e:
        print(f"Error clearing vector store: {e}")
        print("Consider manual deletion of the collection contents.")

def get_table_docs_from_json(json_path: str, run_id: str):
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

        metadata = {"table_name": table_name}
        for key in allowed_metadata_keys:
            if key in table:
                metadata[key] = table[key]

        metadata["column_names"] = column_names
        metadata["type"] = "table"
        metadata["source"] = f"{json_path}_{run_id}"

        table_docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    return table_docs

def get_column_docs_from_json(json_path: str, run_id: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    column_docs = []

    for table in data["tables"]:
        for col in table.get("columns", []):
            column_name = col.get("name", "")
            column_description = col.get("description", "")
            table_name = table.get("name", "")

            page_content = (
                f"Column: {column_name}\n"
                f"COLUMN_NAME: {column_name}\n"
                f"COLUMN_ID: {column_name}\n"
                f"TABLE: {table_name}\n\n"
                f"{column_description}"
            )

            metadata = {"column_name": column_name}
            for key in [
                "data_type",
                "column_token_count",
                "column_token_count_with_column_name",
                "column_token_count_with_column_name_and_description_and_dt",
            ]:
                if key in col:
                    metadata[key] = col[key]

            metadata["table_name"] = table_name
            metadata["type"] = "column"
            metadata["source"] = f"{json_path}_{run_id}"

            column_docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    return column_docs

if __name__ == "__main__":
    try:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting ingestion with run ID: {run_id}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        db_metadata_path = os.path.join(project_dir, "database", "db_metadata.json")
        processed_db_path = os.path.join(project_dir, "database", "processed_db.json")

        # Check if processed_db.json exists
        if not os.path.exists(processed_db_path):
            print("processed_db.json not found.")
            generate_first = 'y'
        else:
            generate_first = input("Do you want to regenerate the processed_db.json before ingestion? (y/n): ").strip().lower()
        
        if generate_first == 'y':
            print("Generating processed_db.json...")
            if generate_processed_db(db_metadata_path):
                print("Successfully generated processed_db.json")
            else:
                print("Failed to generate processed_db.json. Check if db_metadata.json exists and is valid.")
                continue_anyway = input("Do you want to continue with ingestion using the existing or partially generated processed_db.json? (y/n): ").strip().lower()
                if continue_anyway != 'y':
                    print("Exiting.")
                    sys.exit(1)

        clear_first = input("Do you want to clear the vector store before ingestion? (y/n): ").strip().lower()
        if clear_first == 'y':
            clear_vector_store()

        json_path = processed_db_path

        print(f"Loading documents from: {json_path}")
        table_docs = get_table_docs_from_json(json_path, run_id)
        column_docs = get_column_docs_from_json(json_path, run_id)

        if not table_docs and not column_docs:
            print("No documents found in the JSON file. Exiting.")
            sys.exit(0)

        if table_docs:
            print("\nSample Table Doc:\n", table_docs[0], "\n")

        if column_docs:
            print("Sample Column Doc:\n", column_docs[0], "\n")

        all_docs = table_docs + column_docs
        print(f"Loaded {len(all_docs)} total documents (tables + columns).")

        record_manager_namespace = f"pgvector/{PGVECTOR_COLLECTION_NAME}"
        record_manager = SQLRecordManager(
            record_manager_namespace, db_url=PGVECTOR_CONNECTION_STRING
        )

        try:
            print(f"Creating schema for Record Manager namespace: {record_manager_namespace}")
            record_manager.create_schema()
            print("Schema creation/check completed.")
        except Exception as e:
            print(f"Error during Record Manager schema creation: {e}")
            sys.exit(1)

        print(f"Starting indexed ingestion with cleanup='full' and source_id_key='source'...")
        try:
            indexing_stats = index(
                all_docs,
                record_manager,
                vector_store,
                cleanup="full",
                source_id_key="source",
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
            print("Verify PostgreSQL server connectivity and check your PGVECTOR_CONNECTION_STRING.")

    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
