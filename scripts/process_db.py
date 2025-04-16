#!/usr/bin/env python3
"""
Database Metadata Processor

This script processes database metadata JSON files into token-counted formats,
utilizing the helper functions from core/helper.py.

Usage:
    python process_db.py --input-path /path/to/metadata.json --output-path /path/to/output.json
    
Options:
    --input-path: Path to the input database metadata JSON file (default: from settings)
    --output-path: Path where the processed database will be saved (default: from settings)
    --model: LLM model to use for token counting (default: "gpt-3.5-turbo")
    --tables: Comma-separated list of table names to process (default: all tables)
    --summary: Display a summary of token counts (default: False)
"""

import json
import sys
import os
import argparse
import time
import logging
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.helper import (
    get_token_count_for_text,
    get_column_token_count,
    get_table_token_count,
    get_token_count,
    run_get_token_count
)
from settings.settings import DATABASE_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('db_processor')

def cleanup_output_file(output_path):
    """
    Removes any partially created output file if it exists.
    
    Args:
        output_path: Path to the output file to clean up
    """
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Cleaned up partial output file: {output_path}")
            print(f"Cleaned up partial output file: {output_path}")
    except Exception as e:
        logger.error(f"Failed to clean up output file {output_path}: {str(e)}")
        print(f"Warning: Failed to clean up output file {output_path}: {str(e)}")

def display_summary(processed_db):
    """
    Display a summary of the token counts from the processed database
    """
    print("\n" + "="*50)
    print(f"DB TOKEN COUNT SUMMARY FOR: {processed_db.get('db_name', 'N/A')}")
    print("="*50)
    print(f"Total tables: {processed_db.get('table_count', 0)}")
    print(f"Total columns: {processed_db.get('column_count', 0)}")
    print(f"Database token count: {processed_db.get('database_token_count', 0)}")
    print(f"All tables token count: {processed_db.get('tables_token_count', 0)}")
    print(f"Tables with column names token count: {processed_db.get('tables_token_count_with_columns_name', 0)}")
    print(f"Complete token count: {processed_db.get('tables_token_with_columns_name_and_description_and_dt', 0)}")
    
    # Display per-table summary
    if processed_db.get('tables'):
        print("\nPER-TABLE SUMMARY:")
        print("-"*50)
        print(f"{'Table Name':<30} | {'Columns':<8} | {'Token Count':<12}")
        print("-"*50)
        
        for table in processed_db.get('tables', []):
            print(f"{table.get('name', 'N/A'):<30} | {table.get('column_count', 0):<8} | {table.get('table_token_with_columns_name_and_description_and_dt', 0):<12}")
    
    print("="*50)

def process_specific_tables(input_db_path, tables_list, output_path, model="gpt-3.5-turbo"):
    """
    Process only specific tables from the database metadata
    
    Args:
        input_db_path: Path to the input database metadata JSON file
        tables_list: List of table names to process
        output_path: Path where the output will be saved
        model: Model to use for token counting
    
    Returns:
        Dictionary containing the processed data for the specified tables
    """
    try:
        # Load the input database file
        with open(input_db_path, "r") as infile:
            initial_db = json.load(infile)
        
        # Format the input for table token counting
        input_value = [{"table": table_name} for table_name in tables_list]
        
        # Get table token counts for the specified tables
        result = get_table_token_count(initial_db, input_value)
        
        # Create a simplified output format
        output = {
            "db_name": initial_db[0].get("db_name", ""),
            "db_description": initial_db[0].get("db_description", ""),
            "table_count": len(result),
            "tables": result
        }
        
        # Save the processed data
        with open(output_path, "w") as outfile:
            json.dump(output, outfile, indent=2)
            
        return output
        
    except Exception as e:
        # Clean up any partial output
        cleanup_output_file(output_path)
        logger.error(f"Error processing specific tables: {str(e)}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Failed to process specific tables: {str(e)}")

def main():
    """
    Main entry point for the database metadata processor
    """
    parser = argparse.ArgumentParser(description="Process database metadata into token-counted format")
    parser.add_argument("--input-path", help="Path to the input database metadata JSON file")
    parser.add_argument("--output-path", help="Path where the processed database will be saved")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model to use for token counting")
    parser.add_argument("--tables", help="Comma-separated list of table names to process (default: all tables)")
    parser.add_argument("--summary", action="store_true", help="Display a summary of token counts")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Use settings if paths not provided
    input_path = args.input_path if args.input_path else DATABASE_SETTINGS["input_db_path"]
    output_path = args.output_path
    
    # Validate input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} not found.")
        print(f"Error: Input file {input_path} not found.")
        return 1
    
    # Determine output path early so we can clean it up in case of failure
    if not output_path:
        if args.tables:
            output_dir = os.path.dirname(input_path)
            output_path = os.path.join(output_dir, "processed_specific_tables.json")
        else:
            output_path = DATABASE_SETTINGS["output_path"]["processed_db"]
    
    try:
        start_time = time.time()
        logger.info(f"Starting database metadata processing...")
        print(f"Starting database metadata processing...")
        print(f"Input file: {input_path}")
        print(f"Output will be saved to: {output_path}")
        
        # Process specific tables if provided
        if args.tables:
            tables_list = [t.strip() for t in args.tables.split(",")]
            logger.info(f"Processing specific tables: {', '.join(tables_list)}")
            print(f"Processing specific tables: {', '.join(tables_list)}")
            
            result = process_specific_tables(input_path, tables_list, output_path, args.model)
        else:
            # Process the entire database
            if output_path != DATABASE_SETTINGS["output_path"]["processed_db"]:
                # If custom output path provided, we need to temporarily modify settings
                original_output_path = DATABASE_SETTINGS["output_path"]["processed_db"]
                DATABASE_SETTINGS["output_path"]["processed_db"] = output_path
                
                try:
                    result = run_get_token_count(input_path)
                except Exception as e:
                    # Clean up and propagate the error
                    cleanup_output_file(output_path)
                    raise
                finally:
                    # Restore original setting
                    DATABASE_SETTINGS["output_path"]["processed_db"] = original_output_path
            else:
                # Use default output path from settings
                try:
                    result = run_get_token_count(input_path)
                except Exception as e:
                    # Clean up and propagate the error
                    cleanup_output_file(output_path)
                    raise
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {output_path}")
        
        if args.summary:
            display_summary(result)
        
        return 0
        
    except Exception as e:
        # Clean up any partial output file
        cleanup_output_file(output_path)
        
        logger.error(f"Error processing database metadata: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"ERROR: Processing failed: {str(e)}")
        print("Any partial output has been deleted for clean future runs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())