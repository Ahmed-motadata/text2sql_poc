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
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
 
def get_openai_embed_3_small():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
 
def get_openai_embed_3_large():
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
 
def get_jina_embed_base():
    return JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name="jina-embeddings-v2-base-en"
    )

"""