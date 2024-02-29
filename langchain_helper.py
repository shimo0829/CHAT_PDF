import gradio as gr

from langchain_community.document_loaders import PyPDFLoader

from langchain.indexes.vectorstore import VectorstoreIndexCreator

from dotenv import load_dotenv

load_dotenv()

file_path = "./electronics.pdf"
local_persist_path = "./vector_store"

loader = PyPDFLoader(file_path)
index = VectorstoreIndexCreator().from_loaders([loader])
index.vectorstore.persist() 
  
def answer_question(question):
    ans = index.query_with_sources(question, chain_type="map_reduce")
    return ans['answer']

iface = gr.Interface(fn=answer_question, inputs="text", outputs="text", title="Interactive PDF", description="Enter your question in Chinese.")
iface.launch()