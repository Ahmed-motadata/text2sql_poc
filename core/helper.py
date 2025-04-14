import tiktoken
import json
import sys
import os
import argparse

# Add the parent directory to the path to allow imports from settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.settings import DATABASE_SETTINGS

def get_token_count_for_text(text, model="gpt-3.5-turbo"):
    """
    Returns the token count for a given text using the specified model's encoding.
    
    :param text: Text to tokenize.
    :param model: Model name for encoding (default "gpt-3.5-turbo").
    :return: Integer representing the token count.
    """
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

def get_column_token_count(initial_db, input_value):
    """
    Processes column token counts for the given initial_db based on input_value.
    
    Three modes:
      1) If input_value is empty ([]), processes all tables and all columns.
      2) If input_value contains table names with "column" set as None, processes all columns for those tables.
      3) If input_value contains table names and a list of columns, processes only the specified columns.
    
    For each column, computes:
      - token_count_with_columns_name: Token count of the column name.
      - token_count_with_columns_name_description_dt: Token count of "column name + column description + column datatype".
    
    The output for each table is formatted so that table-level details come first followed by the list of columns.
    
    Finally, the function writes the output (a list of dictionaries) into a JSON file.
    
    :param initial_db: The full database metadata (a list of database dictionaries).
    :param input_value: List defining specific table and column processing.
    :return: The output as a list of processed table token counts.
    """
    
    processed_results = []
    
    def process_table(table, selected_columns=None):
        aggregated_token_name = 0
        aggregated_token_combined = 0
        columns_output = []
        
        # Determine which columns to process:
        if selected_columns is None:
            columns = table.get("columns", [])
        else:
            columns = [col for col in table.get("columns", []) if col.get("name") in selected_columns]
        
        for col in columns:
            col_name = col.get("name", "")
            col_desc = col.get("description", "")
            col_type = col.get("type", "")
            token_name = get_token_count_for_text(col_name)
            # Construct combined string for token count.
            combined_text = f"{col_name} {col_desc} {col_type}"
            token_combined = get_token_count_for_text(combined_text)
            aggregated_token_name += token_name
            aggregated_token_combined += token_combined
            columns_output.append({
                "name": col_name,
                "token_count_with_columns_name": token_name,
                "token_count_with_columns_name_description_dt": token_combined
            })
        
        return {
            "table_name": table.get("name", ""),
            "token_count_with_columns_name": aggregated_token_name,
            "token_count_with_columns_name_description_dt": aggregated_token_combined,
            "column_count": len(columns),
            "columns": columns_output
        }
    
    if not input_value:
        # Process all tables across all databases.
        for db in initial_db:
            for table in db.get("tables", []):
                processed_results.append(process_table(table, selected_columns=None))
    else:
        # Process only specified tables and columns.
        for item in input_value:
            table_name = item.get("table")
            selected_columns = item.get("column")
            for db in initial_db:
                for table in db.get("tables", []):
                    if table.get("name") == table_name:
                        processed_results.append(process_table(table, selected_columns))
    
    # Use the path from DATABASE_SETTINGS for output
    output_path = DATABASE_SETTINGS["output_path"]["column_token_count"]
    with open(output_path, "w") as outfile:
        json.dump(processed_results, outfile, indent=2)
    
    return processed_results

def get_table_token_count(initial_db, input_value):
    """
    Computes table-level token counts based on the provided initial_db and input_value.
    
    Input parameters:
      - initial_db: The entire database metadata loaded from a JSON file.
      - input_value: A list of dicts where each dict has a "table" key indicating the table to process.
                   Example:
                   [
                     {"table": "table1"},
                     {"table": "table2"}
                   ]
    
    Processing:
      - If input_value is empty ([]), process all tables in the db.
      - Otherwise, process only the specified tables.
      - For each table, compute:
           token_count: token count for "table name + table description" 
           column_count: number of columns in the table, as returned by get_column_token_count.
           token_columns_with_columns_name: table token_count plus the aggregated token count for column names.
           token_with_columns_name_and_description_and_dt: table token_count plus the aggregated token count for
               "column name + column description + column datatype" across all columns.
    
    The output for each table is a dictionary with keys:
      - table_name
      - column_count
      - token_count
      - token_columns_with_columns_name
      - token_with_columns_name_and_description_and_dt
    
    The final output (a list of such dictionaries) is stored in a JSON file.
    
    :param initial_db: The full database metadata.
    :param input_value: A list of table specifications; if empty, process the entire db.
    :return: A list of dictionaries with the computed table-level token counts.
    """
    results = []

    def process_table(table):
        # Compute token count for the table (table name + table description)
        table_name = table.get("name", "")
        table_description = table.get("description", "")
        table_token_count = get_token_count_for_text(f"{table_name} {table_description}")

        # Prepare input for column-level processing: using column as None to process all columns.
        col_input = [{"table": table_name, "column": None}]
        col_output = get_column_token_count(initial_db, col_input)
        # We expect the returned list to contain the entry for this table.
        if col_output:
            col_data = col_output[0]
        else:
            col_data = {"column_count": 0, "token_count_with_columns_name": 0,
                        "token_count_with_columns_name_description_dt": 0}

        # Calculate the final values by adding table-level token count and column-level token counts.
        token_columns_with_columns_name = table_token_count + col_data["token_count_with_columns_name"]
        token_with_columns_name_and_description_and_dt = table_token_count + col_data["token_count_with_columns_name_description_dt"]

        return {
            "table_name": table_name,
            "column_count": col_data["column_count"],
            "token_count": table_token_count,
            "token_columns_with_columns_name": token_columns_with_columns_name,
            "token_with_columns_name_and_description_and_dt": token_with_columns_name_and_description_and_dt
        }

    # Determine which tables to process.
    if not input_value:  # Process all tables if input_value is empty.
        for db in initial_db:
            for table in db.get("tables", []):
                results.append(process_table(table))
    else:
        # Process only the specified tables.
        table_names_to_process = {item.get("table") for item in input_value}
        for db in initial_db:
            for table in db.get("tables", []):
                if table.get("name") in table_names_to_process:
                    results.append(process_table(table))

    # Write results into a JSON file using the path from DATABASE_SETTINGS
    output_path = DATABASE_SETTINGS["output_path"]["table_token_count"]
    with open(output_path, "w") as outfile:
        json.dump(results, outfile, indent=2)

    return results

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
    
    The final output is a dictionary in the format described in the docstring and is stored in two JSON files:
    token_count.json and processed_db.json.
    
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
            col_type = col.get("data_type", "")
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
                "column_token_count": token_count,
                "column_token_count_with_column_name": token_count_with_column_name,
                "column_token_count_with_column_name_and_description_and_dt": token_count_with_column_name_and_desc_dt
            })
        
        # For the table-level output, add the table's own token count
        # to the aggregated columns' tokens.
        token_columns_with_columns_name = table_token_count + agg_token_columns_with_name
        token_with_columns_name_and_description_and_dt = table_token_count + agg_token_columns_with_name_and_desc_dt
        
        processed_tables.append({
            "name": table_name,
            "description": table_description,
            "column_count": len(cols),
            "table_token_count": table_token_count,
            "table_token_columns_with_columns_name": token_columns_with_columns_name,
            "table_token_with_columns_name_and_description_and_dt": token_with_columns_name_and_description_and_dt,
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
    
    # Write the processed database structure to a JSON file using the path from DATABASE_SETTINGS
    token_count_path = DATABASE_SETTINGS["output_path"]["token_count"]
    with open(token_count_path, "w") as outfile:
        json.dump(processed_db, outfile, indent=2)
    
    # Also save to processed_db.json
    processed_db_path = DATABASE_SETTINGS["output_path"]["processed_db"]
    with open(processed_db_path, "w") as outfile:
        json.dump(processed_db, outfile, indent=2)
    
    return processed_db

# Main functions to run each token counting process
def run_get_token_count(input_db_path=None):
    """Helper function to run the get_token_count process"""
    # Use the provided path or get it from settings
    if input_db_path is None:
        input_db_path = DATABASE_SETTINGS["input_db_path"]
        
    with open(input_db_path, "r") as infile:
        initial_db = json.load(infile)
    
    result = get_token_count(initial_db)
    print(f"Processed DB token counts stored in {DATABASE_SETTINGS['output_path']['processed_db']}")
    return result

def run_get_table_token_count(input_db_path=None, input_value=None):
    """Helper function to run the get_table_token_count process"""
    # Use the provided path or get it from settings
    if input_db_path is None:
        input_db_path = DATABASE_SETTINGS["input_db_path"]
        
    with open(input_db_path, "r") as infile:
        initial_db = json.load(infile)
    
    # Default to empty list if not provided
    if input_value is None:
        input_value = []
        
    result = get_table_token_count(initial_db, input_value)
    print(f"Table-level token count processed and stored in {DATABASE_SETTINGS['output_path']['table_token_count']}")
    return result

def run_get_column_token_count(input_db_path=None, input_value=None):
    """Helper function to run the get_column_token_count process"""
    # Use the provided path or get it from settings
    if input_db_path is None:
        input_db_path = DATABASE_SETTINGS["input_db_path"]
        
    with open(input_db_path, "r") as infile:
        initial_db = json.load(infile)
    
    # Default to empty list if not provided
    if input_value is None:
        input_value = []
        
    result = get_column_token_count(initial_db, input_value)
    print(f"Column-level token counts stored in {DATABASE_SETTINGS['output_path']['column_token_count']}")
    return result

# This allows each function to be run individually if the helper module is executed directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process token counts for database metadata")
    parser.add_argument("--function", choices=["token", "table", "column"], 
                      help="Function to run: token (get_token_count), table (get_table_token_count), column (get_column_token_count)")
    
    args = parser.parse_args()
    
    if args.function == "token":
        run_get_token_count()
    elif args.function == "table":
        run_get_table_token_count()
    elif args.function == "column":
        run_get_column_token_count()
    else:
        # Run all three by default
        print("Running all token counting functions...")
        run_get_column_token_count()
        run_get_table_token_count()
        run_get_token_count()
        print("All token counting operations completed successfully.")