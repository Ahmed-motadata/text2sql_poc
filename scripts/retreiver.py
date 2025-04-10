import sys
import os
 
# Add the 'base' folder to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "base")))
 
<<<<<<< HEAD
from vector_store import vector_store
=======
from vector_store_sid import vector_store
>>>>>>> Sid
 
results = vector_store.similarity_search("SHow me request table?", k=1)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)