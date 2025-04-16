import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
 
load_dotenv()
 
OPENAI_EMBED_ADA = OpenAIEmbeddings(model="text-embedding-ada-002")
OPENAI_EMBED_3_SMALL = OpenAIEmbeddings(model="text-embedding-3-small")
OPENAI_EMBED_3_LARGE = OpenAIEmbeddings(model="text-embedding-3-large")
 
JINA_EMBED_BASE_V2 = JinaEmbeddings(model="jina-embeddings-v2-base-en", api_key=os.getenv("JINA_API_KEY"))
JINA_EMBED_CLIP_V2 = JinaEmbeddings(model="jina-clip-v2", api_key=os.getenv("JINA_API_KEY"))
JINA_EMBED_SMALL_V2 = JinaEmbeddings(model="jina-embeddings-v2-small-en", api_key=os.getenv("JINA_API_KEY"))
JINA_EMBED_V3 = JinaEmbeddings(model="jina-embeddings-v3", api_key=os.getenv("JINA_API_KEY"))