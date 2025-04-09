import tiktoken
import json
from get_column_token_count import get_column_token_count  # Import the function from the separate file

def get_token_count_for_text(text, model="gpt-3.5-turbo"):
    """
    Returns the token count for a given text using the specified model's encoding.
    
    :param text: Text to tokenize.
    :param model: Model name for encoding (default is "gpt-3.5-turbo").
    :return: Integer representing the token count.
    """
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

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
    
    The final output (a list of such dictionaries) is stored in a JSON file ("get_table_token_count.json").
    
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

    # Write results into a JSON file.
    with open("/home/ahmedraza/genBI/Text2SQL/database/get_table_token_count.json", "w") as outfile:
        json.dump(results, outfile, indent=2)

    return results

# Example usage:
if __name__ == "__main__":
    # Load initial_db from a JSON file.
    with open("/home/ahmedraza/genBI/Text2SQL/database/db_metadata.json", "r") as infile:
        initial_db = json.load(infile)
    
    # Define input_value.
    get_token_count_input = []
    
    # To process the entire db, use an empty list: []
    # get_token_count_input = []

    output = get_table_token_count(initial_db, get_token_count_input)
    
    print("Table-level token count processed and stored in get_table_token_count.json")
