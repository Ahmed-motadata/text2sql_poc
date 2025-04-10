import sys
import os
 
# Add the 'base' folder to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "base")))
 
from vector_store import vector_store
 
results = vector_store.similarity_search("SHow me request table?", k=1)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)