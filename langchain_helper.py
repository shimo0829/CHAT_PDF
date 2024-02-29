from langchain_community.document_loaders import PyPDFLoader

from langchain.indexes.vectorstore import VectorstoreIndexCreator

from dotenv import load_dotenv

load_dotenv()

file_path = "./electronics.pdf"

local_persist_path = "./vector_store"

loader = PyPDFLoader(file_path)

index = VectorstoreIndexCreator().from_loaders([loader])

index.vectorstore.persist() 

ans = index.query_with_sources("半波整流的相關公式有哪些?", chain_type="map_reduce")

print(ans)   