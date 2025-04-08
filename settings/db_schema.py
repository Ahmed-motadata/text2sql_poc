import os
import psycopg2
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def establish_connection(dbname=None, user=None, password=None, host=None, port=None):
    """
    Establish a connection to the PostgreSQL database.
    
    Args:
        dbname: Database name (defaults to DB_NAME from env)
        user: Database user (defaults to DB_USER from env)
        password: Database password (defaults to DB_PASSWORD from env)
        host: Database host (defaults to DB_HOST from env or 'localhost')
        port: Database port (defaults to DB_PORT from env or '5432')
    
    Returns:
        tuple: (connection, cursor) if successful, (None, None) on failure
    """
    # Use environment variables if parameters are not provided
    dbname = dbname or os.environ.get('DB_NAME')
    user = user or os.environ.get('DB_USER')
    password = password or os.environ.get('DB_PASSWORD')
    host = host or os.environ.get('DB_HOST', 'localhost')
    port = port or os.environ.get('DB_PORT', '5432')
    
    try:
        # Attempt to connect to the specified database
        connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        cursor = connection.cursor()
        
        # Print verification message
        print(f"✅ Successfully connected to PostgreSQL database '{dbname}' at {host}:{port}")
        logging.info(f"Connected to PostgreSQL database: {dbname}")
        
        return connection, cursor
    
    except psycopg2.OperationalError as error:
        # If the database doesn't exist, provide helpful guidance
        if "database" in str(error) and "does not exist" in str(error):
            print(f"❌ Database '{dbname}' does not exist.")
            print(f"\nTo create the database, connect to PostgreSQL and run:")
            print(f"    CREATE DATABASE {dbname};\n")
        else:
            print(f"❌ Failed to connect to the database: {error}")
        
        logging.error(f"Error connecting to PostgreSQL database: {error}")
        return None, None
    
    except Exception as error:
        print(f"❌ Failed to connect to the database: {error}")
        logging.error(f"Error connecting to PostgreSQL database: {error}")
        return None, None

def close_connection(connection, cursor):
    """
    Close the database connection and cursor.
    
    Args:
        connection: The database connection to close
        cursor: The cursor to close
    """
    if cursor:
        cursor.close()
    if connection:
        connection.close()
        print("Database connection closed")
        logging.info("PostgreSQL connection closed")

def keep_connection_alive(connection, cursor, interval=30):
    """
    Keep the database connection alive by running a simple query at regular intervals.
    This function runs in the background and doesn't produce any output.
    
    Args:
        connection: The database connection to keep alive
        cursor: The cursor for the connection
        interval: Number of seconds between keep-alive queries (default: 30)
    
    Returns:
        A callable that when called will stop the keep-alive process
    """
    import threading
    import time
    
    stop_event = threading.Event()
    
    def _keep_alive_worker():
        while not stop_event.is_set():
            try:
                # Run a minimal query to keep the connection active
                cursor.execute("SELECT 1")
                cursor.fetchone()
            except:
                # Silently handle any errors
                pass
            
            # Sleep for the specified interval
            time.sleep(interval)
    
    # Start the keep-alive thread
    keep_alive_thread = threading.Thread(target=_keep_alive_worker, daemon=True)
    keep_alive_thread.start()
    
    # Return a function to stop the keep-alive process
    def stop_keep_alive():
        stop_event.set()
        if keep_alive_thread.is_alive():
            keep_alive_thread.join(timeout=1.0)
    
    return stop_keep_alive

# Example usage
if __name__ == "__main__":
    connection, cursor = establish_connection()
    
    if connection:
        try:
            # Test a simple query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"PostgreSQL version: {version[0]}")
            
            # Start keep-alive and wait indefinitely
            print("\nKeeping connection open - press Ctrl+C to exit")
            
            # Start the keep-alive process
            stop_keep_alive = keep_connection_alive(connection, cursor)
            
            # Keep the main thread running
            import signal
            import time
            
            # Define signal handler for graceful exit
            def signal_handler(sig, frame):
                print("\nExiting and closing database connection...")
                stop_keep_alive()
                close_connection(connection, cursor)
                exit(0)
            
            # Register signal handler for Ctrl+C
            signal.signal(signal.SIGINT, signal_handler)
            
            # Keep the main thread alive
            while True:
                time.sleep(60)  # Just sleep and let the keep-alive thread do its work
                
        except Exception as e:
            print(f"Error: {e}")
            close_connection(connection, cursor)
    else:
        print("Failed to establish database connection.")
