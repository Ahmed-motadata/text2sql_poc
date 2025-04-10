import sys
import os
 
# Add the parent directory to the path to allow imports from base
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
# Import the vector_store module correctly
from base.vector_store import vector_store
 
results = vector_store.similarity_search("SHow me request table's columns?", k=1)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)