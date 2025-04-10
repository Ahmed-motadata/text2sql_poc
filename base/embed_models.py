import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings

# Load environment variables
load_dotenv()

# Use functions instead of initializing everything on import

def get_openai_embed_ada():
    """
    Returns an instance of OpenAIEmbeddings using the Ada model.
    Only created when this function is called.
    """
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def get_openai_embed_3_small():
    """
    Returns an instance of OpenAIEmbeddings using the small text-embedding-3 model.
    Only created when this function is called.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def get_openai_embed_3_large():
    """
    Returns an instance of OpenAIEmbeddings using the large text-embedding-3 model.
    Only created when this function is called.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def get_jina_embed_base():
    """
    Returns an instance of JinaEmbeddings using the base English model.
    Only created when this function is called.
    """
    return JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name="jina-embeddings-v2-base-en"
    )

def get_jina_embed_clip():
    """
    Returns an instance of JinaEmbeddings using the CLIP model.
    Only created when this function is called.
    """
    return JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name="jina-clip-v2"
    )

def get_jina_embed_small():
    """
    Returns an instance of JinaEmbeddings using the small English model.
    Only created when this function is called.
    """
    return JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name="jina-embeddings-v2-small-en"
    )

# For backward compatibility (if any existing code relies on these variables)
# These will be deprecated in favor of the function calls
openai_embed_ada = get_openai_embed_ada()
openai_embed_3_small = get_openai_embed_3_small()
openai_embed_3_large = get_openai_embed_3_large()
jina_embed_base = get_jina_embed_base()
jina_embed_clip = get_jina_embed_clip()
jina_embed_small = get_jina_embed_small()