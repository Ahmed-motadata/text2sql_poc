import json
import sys
import os
from langchain_core.documents import Document

# Add the parent directory to the path to allow imports from base and settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the vector_store properly
from base.vector_store import vector_store  

def get_table_docs_from_json(json_path: str):
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
        # Step 1: Extract column names
        column_names = [col["name"] for col in table.get("columns", [])]

        # Step 2: Prepare content
        table_name = table.get("name", "")
        description = table.get("description", "")
        page_content = f"Table: {table_name}\n\n{description}"

        # Step 3: Filter metadata
        metadata = {}
        for key in allowed_metadata_keys:
            if key in table:
                metadata[key] = table[key]
            else:
                print(f"[⚠️ Missing] '{key}' not found in table '{table_name}'")

        # Step 4: Add additional metadata
        metadata["column_names"] = column_names
        metadata["type"] = "table"

        # Step 5: Create Document
        table_docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    return table_docs


def get_column_docs_from_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    column_docs = []

    for table in data["tables"]:
        for col in table.get("columns", []):
            # Step 1: Page content
            column_name = col.get("name", "")
            column_description = col.get("description", "")
            page_content = f"Column: {column_name}\n\n{column_description}"

            # Step 2: Metadata
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

            # Step 3: Create Document
            column_docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    return column_docs


if __name__ == "__main__":
    try:
        # Use absolute path for better reliability
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        json_path = os.path.join(project_dir, "database", "processed_db.json")

        table_docs = get_table_docs_from_json(json_path)
        column_docs = get_column_docs_from_json(json_path)

        print("\nSample Table Doc:\n", table_docs[0], "\n")
        print("Sample Column Doc:\n", column_docs[0], "\n")

        all_docs = table_docs + column_docs
        print(f"Loaded {len(all_docs)} total documents (tables + columns)... Ingesting into vector store...")

        try:
            vector_store.add_documents(all_docs)
            print("Ingestion completed successfully.")
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            print("Make sure the PostgreSQL server is running and the credentials are correct.")
            print("Check your .env file for proper database configuration.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Please check that the processed_db.json file exists.")
