import psycopg2
from psycopg2 import pool
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PostgresConnection:
    """
    A class for managing connections to PostgreSQL database
    """
    
    def __init__(self, dbname=None, user=None, password=None, host=None, port=None):
        """Initialize the database connection parameters"""
        # Use environment variables if parameters are not provided
        self.dbname = dbname or os.environ.get('DB_NAME')
        self.user = user or os.environ.get('DB_USER')
        self.password = password or os.environ.get('DB_PASSWORD')
        self.host = host or os.environ.get('DB_HOST', 'localhost')
        self.port = port or os.environ.get('DB_PORT', '5432')
        self.connection = None
        self.cursor = None
        self.connection_pool = None
    
    @classmethod
    def from_env(cls):
        """Create a connection instance using environment variables"""
        return cls()
        
    def connect(self):
        """Establish a connection to the PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.cursor = self.connection.cursor()
            logging.info("Connected to PostgreSQL database successfully")
            return True
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Error connecting to PostgreSQL database: {error}")
            return False
            
    def create_connection_pool(self, min_conn=1, max_conn=10):
        """Create a connection pool for better performance in multi-threaded apps"""
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            logging.info("PostgreSQL connection pool created successfully")
            return True
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Error creating connection pool: {error}")
            return False
    
    def execute_query(self, query, params=None):
        """Execute a query and return the results"""
        try:
            if not self.connection or self.connection.closed:
                self.connect()
                
            self.cursor.execute(query, params)
            
            if query.strip().upper().startswith(("SELECT", "SHOW", "DESCRIBE")):
                return self.cursor.fetchall()
            else:
                self.connection.commit()
                return True
                
        except (Exception, psycopg2.Error) as error:
            logging.error(f"Error executing query: {error}")
            self.connection.rollback()
            return False
            
    def get_table_schema(self, table_name):
        """Get the schema of a specific table"""
        query = """
        SELECT column_name, data_type 
        FROM information_schema.columns
        WHERE table_name = %s
        """
        return self.execute_query(query, (table_name,))
    
    def get_all_tables(self):
        """Get a list of all tables in the database"""
        query = """
        SELECT table_name 
        FROM information_schema.tables
        WHERE table_schema = 'public'
        """
        return self.execute_query(query)
        
    def close(self):
        """Close the database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logging.info("PostgreSQL connection closed")
            
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()
