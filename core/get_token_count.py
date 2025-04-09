import tiktoken
import json
from get_table_token_count import get_table_token_count
from get_column_token_count import get_column_token_count

def get_token_count_for_text(text, model="gpt-3.5-turbo"):
    """
    Returns the token count for a given text using the specified model's encoding.
    
    :param text: Text to tokenize.
    :param model: Model name for encoding (default "gpt-3.5-turbo").
    :return: Integer representing the token count.
    """
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

def get_token_count(initial_db):
    """
    Processes the entire database metadata (initial_db) and produces overall token usage.
    
    Processing includes:
      - Calling get_table_token_count with an empty input list so that it processes all tables.
      - Calling get_column_token_count similarly for column-level details.
      - Computing:
            * table_count: Number of tables in the db.
            * column_count: Sum of columns across all tables.
            * database_token_count: Token count of (db_name + db_description)
            * tables_token_count: Sum of token counts for each table (table name + description)
            * tables_token_count_with_columns_name: Sum of (table token count + aggregated token count for column names)
            * tables_token_with_columns_name_and_description_and_dt: Sum of (table token count + aggregated token count for 
              (column name + column description + column datatype))
      - Processing each table to include detailed column information.
    
    The final output is a dictionary in the format:
    
      {
         "db_name": ...,
         "db_description": ...,
         "table_count": ...,
         "column_count": ...,
         "database_token_count": ...,
         "tables_token_count": ...,
         "tables_token_count_with_columns_name": ...,
         "tables_token_with_columns_name_and_description_and_dt": ...,
         "tables": [
             {
                "name": ...,
                "description": ...,
                "column_count": ...,
                "token_count": ...,
                "token_columns_with_columns_name": ...,
                "token_with_columns_name_and_description_and_dt": ...,
                "columns": [
                    {
                       "name": ...,
                       "description": ...,
                       "data_type": ...,
                       "token_count": ...,
                       "token_count_with_column_name": ...,
                       "token_count_with_column_name_and_description_and_dt": ...
                    },
                    ...
                ]
             },
             ...
         ]
      }
    
    Finally, the processed database dictionary is stored in "processed_db.json".
    
    :param initial_db: The full database metadata (list of database dictionaries).
    :return: The processed database dictionary.
    """
    # For simplicity, we assume initial_db is a list and process the first database.
    db_meta = initial_db[0]
    db_name = db_meta.get("db_name", "")
    db_description = db_meta.get("db_description", "")
    
    # Get table-level data using our function (processing all tables).
    table_level_data = get_table_token_count(initial_db, [])  # returns a list of table-level aggregates

    # Also get column-level aggregates (if needed) from get_column_token_count.
    # (Though our final output will recompute detailed column info.)
    _ = get_column_token_count(initial_db, [])  # Not used directly below, but illustrates the call.
    
    # Initialize accumulators.
    table_count = 0
    column_count = 0
    tables_token_count = 0
    tables_token_count_with_columns_name = 0
    tables_token_with_columns_name_and_description_and_dt = 0
    processed_tables = []

    # Process each table from the db metadata.
    for table in db_meta.get("tables", []):
        table_count += 1
        table_name = table.get("name", "")
        table_description = table.get("description", "")
        # Compute the table's token count (name + description).
        table_token_count = get_token_count_for_text(f"{table_name} {table_description}")
        
        # Process columns for the table.
        cols = table.get("columns", [])
        column_count += len(cols)
        
        agg_token_columns_with_name = 0
        agg_token_columns_with_name_and_desc_dt = 0
        processed_columns = []
        for col in cols:
            col_name = col.get("name", "")
            col_desc = col.get("description", "")
            col_type = col.get("type", "")
            # Compute token counts for the column. (The logic here can be modified as needed.)
            # token_count: tokens for (column name + column description)
            token_count = get_token_count_for_text(f"{col_name} {col_desc}")
            # token_count_with_column_name: tokens for the column name alone.
            token_count_with_column_name = get_token_count_for_text(col_name)
            # token_count_with_column_name_and_description_and_dt: tokens for (column name + column description + column datatype)
            token_count_with_column_name_and_desc_dt = get_token_count_for_text(f"{col_name} {col_desc} {col_type}")
            
            agg_token_columns_with_name += token_count_with_column_name
            agg_token_columns_with_name_and_desc_dt += token_count_with_column_name_and_desc_dt
            
            processed_columns.append({
                "name": col_name,
                "description": col_desc,
                "data_type": col_type,
                "token_count": token_count,
                "token_count_with_column_name": token_count_with_column_name,
                "token_count_with_column_name_and_description_and_dt": token_count_with_column_name_and_desc_dt
            })
        
        # For the table-level output, add the table's own token count
        # to the aggregated columns' tokens.
        token_columns_with_columns_name = table_token_count + agg_token_columns_with_name
        token_with_columns_name_and_description_and_dt = table_token_count + agg_token_columns_with_name_and_desc_dt
        
        processed_tables.append({
            "name": table_name,
            "description": table_description,
            "column_count": len(cols),
            "token_count": table_token_count,
            "token_columns_with_columns_name": token_columns_with_columns_name,
            "token_with_columns_name_and_description_and_dt": token_with_columns_name_and_description_and_dt,
            "columns": processed_columns
        })
        
        tables_token_count += table_token_count
        tables_token_count_with_columns_name += token_columns_with_columns_name
        tables_token_with_columns_name_and_description_and_dt += token_with_columns_name_and_description_and_dt

    # Compute the database-level token count from the db_name and db_description.
    database_token_count = get_token_count_for_text(f"{db_name} {db_description}")
    
    processed_db = {
        "db_name": db_name,
        "db_description": db_description,
        "table_count": table_count,
        "column_count": column_count,
        "database_token_count": database_token_count,
        "tables_token_count": tables_token_count,
        "tables_token_count_with_columns_name": tables_token_count_with_columns_name,
        "tables_token_with_columns_name_and_description_and_dt": tables_token_with_columns_name_and_description_and_dt,
        "tables": processed_tables
    }
    
    # Write the processed database structure to a JSON file.
    with open("/home/ahmedraza/genBI/Text2SQL/database/get_token_count.json", "w") as outfile:
        json.dump(processed_db, outfile, indent=2)
    
    return processed_db

if __name__ == "__main__":
    # Load the initial_db from a JSON file.
    with open("/home/ahmedraza/genBI/Text2SQL/database/db_metadata.json", "r") as infile:
        initial_db = json.load(infile)
    
    output = get_token_count(initial_db)
    print("Processed DB token counts stored in processed_db.json")
