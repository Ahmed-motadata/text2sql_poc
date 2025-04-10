import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
 
# Load environment variables
load_dotenv()

# OpenAI embeddings
openai_embed_ada = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

openai_embed_3_small = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

openai_embed_3_large = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Jina embeddings
jina_embed_base = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v2-base-en"
)

jina_embed_clip = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-clip-v2"
)

jina_embed_small = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v2-small-en"
)


"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
 
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

"""