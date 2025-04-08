from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

def connect_to_qdrant(
    host="localhost",
    port=6333,
    grpc_port=None,
    api_key=None,
    https=False,
    prefer_grpc=False,
    timeout=None,
    **kwargs
):
    """
    Establishes a connection to the Qdrant vector store.
    
    Args:
        host (str): Hostname or IP address of the Qdrant server. Default is localhost.
        port (int): HTTP port of the Qdrant server. Default is 6333.
        grpc_port (int, optional): gRPC port of the Qdrant server if different from HTTP port.
        api_key (str, optional): API key for Qdrant Cloud authentication.
        https (bool): Whether to use HTTPS protocol. Default is False.
        prefer_grpc (bool): Whether to use gRPC protocol. Default is False.
        timeout (float, optional): Timeout for connection in seconds.
        **kwargs: Additional arguments to pass to the QdrantClient constructor.
        
    Returns:
        QdrantClient: An instance of the Qdrant client connected to the server.
        
    Raises:
        Exception: Any error that occurs during connection attempt.
    """
    try:
        logging.info(f"Connecting to Qdrant at {host}:{port}")
        client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            https=https,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
            **kwargs
        )
        # Test the connection
        client.get_collections()
        logging.info("Successfully connected to Qdrant")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant: {e}")
        raise
