import json
from langchain_core.documents import Document
import sys
import os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "base")))
<<<<<<< HEAD
from vector_store_sid import vector_store  
=======
from vector_store import vector_store  
>>>>>>> Sid
 
from langchain_core.documents import Document
import json
 
def get_table_docs_from_json(json_path: str):
    import json
    from langchain_core.documents import Document  # or from langchain.schema if you're on older version
 
    with open(json_path, "r") as f:
        data = json.load(f)
 
    table_docs = []
 
    for table in data["tables"]:
        # Step 1: Extract column list
        column_names = [col["name"] for col in table.get("columns", [])]
 
        # Step 2: Prepare page_content (table name + table description)
        table_name = table.get("name", "")
        description = table.get("description", "")
        page_content = f"Table: {table_name}\n\n{description}"
 
        # Step 3: Extract metadata with only required fields
        allowed_metadata_keys = {
            "column_count",
            "token_count",
            "token_columns_with_columns_name",
            "token_with_columns_name_and_description_and_dt"
        }
 
        metadata = {
            k: v for k, v in table.items()
            if k in allowed_metadata_keys
        }
 
        # Step 4: Add column list
        metadata["column_names"] = column_names
 
        # Step 5: Append to table_docs
        table_docs.append(Document(page_content=page_content.strip(), metadata=metadata))
 
    return table_docs
<<<<<<< HEAD

def get_column_docs_from_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    column_docs = []

    for table in data["tables"]:
        for col in table.get("columns", []):
            # Step 1: Build page content
            column_name = col.get("name", "")
            column_description = col.get("description", "")
            page_content = f"Column: {column_name}\n\n{column_description}"

            # Step 2: Metadata
            metadata = {}
            for key in [
                "data_type",
                "token_count",
                "token_count_with_column_name",
                "token_count_with_column_name_and_description_and_dt"
            ]:
                if key in col:
                    metadata[key] = col[key]

            # Step 3: Optionally add table name for better traceability
            metadata["table_name"] = table.get("name", "")

            # Step 4: Append Document
            column_docs.append(Document(page_content=page_content.strip(), metadata=metadata))

    return column_docs
=======
>>>>>>> Sid
 
 
 
 
 
if __name__ == "__main__":
<<<<<<< HEAD
    docs = get_table_docs_from_json("/home/siddhi/Documents/Siddhi/Text2SQL/text2sql_poc/database/processed_db.json")
=======
    docs = get_table_docs_from_json("/home/ahmedraza/genBI/Text2SQL/database/processed_db.json")
>>>>>>> Sid
    
    print(f"Loaded {len(docs)} table documents... Ingesting into vector store...")
 
    vector_store.add_documents(docs)
 
    print("Ingestion completed.")
 