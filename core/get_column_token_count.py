import tiktoken
import json

def get_token_count_for_text(text, model="gpt-3.5-turbo"):
    """
    Returns the token count for a given text using the specified model's encoding.
    
    :param text: Text to tokenize.
    :param model: Model name for encoding (default "gpt-3.5-turbo").
    :return: Integer token count.
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
    
    Finally, the function writes the output (a list of dictionaries) into a JSON file ("output.json").
    
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
                "column_token_count_with_columns_name": token_name,
                "column_token_count_with_columns_name_description_dt": token_combined
            })
        
        return {
            "table_name": table.get("name", ""),
            "table_token_count_with_columns_name": aggregated_token_name,
            "table_token_count_with_columns_name_description_dt": aggregated_token_combined,
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
    
    with open("/home/ahmedraza/genBI/Text2SQL/database/get_column_token_count.json", "w") as outfile:
        json.dump(processed_results, outfile, indent=2)
    
    return processed_results

# Example usage:
if __name__ == "__main__":
    # Load initial_db from a JSON file.
    # Ensure that "initial_db.json" exists in the same directory with proper structure.
    with open("/home/ahmedraza/genBI/Text2SQL/database/db_metadata.json", "r") as infile:
        initial_db = json.load(infile)
    
    # Define the input value for processing. You can modify this as required.
    get_token_count_input = [
        {
            "table": "request",
            "column": ["id", "name"]
        },
        {
            "table": "flotouser", 
            "column": ["id", "name"]
        }
    ]
    
    # Call get_column_token_count using the loaded initial_db and the input definition.
    output = get_column_token_count(initial_db, get_token_count_input)
    
    # The output is stored in "output.json" and also returned in the variable 'output'.
