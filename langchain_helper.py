'''
PyPDFLoader : 用於加載PDF文件
VectorstoreIndexCreator : 用於建立文件的向量索引
load_dotenv : 用於加載環境變量

'''
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader          
from langchain.indexes.vectorstore import VectorstoreIndexCreator     
from dotenv import load_dotenv     

load_dotenv()  #加載環境變量(OPENAI_API_KEY)    

file_path = "./electronics.pdf"  #定義PDF文件路徑
local_persist_path = "./vector_store"  #定義向量存儲路徑

loader = PyPDFLoader(file_path)  #建立PyPDFLoader用來加載PDF文件的路徑
index = VectorstoreIndexCreator().from_loaders([loader]) #建立向量索引
index.vectorstore.persist()  #向量存儲持久化，提供後續查詢的便利性

def answer_question(question):  
    ans = index.query_with_sources(question, chain_type="map_reduce")  #透過給定的問題進行查詢，並指定查詢類型為"map_reduce"
    answer_text = "Answer: {}\n".format(ans['answer'])  #將答案的轉為字符串的格式，並賦值給'answer_text'變量
    return answer_text  #將答案傳回給使用者

'''
創建Gradio介面
fn → 所要使用的函數
inputs, outputs → 指定輸入與輸出的類型
title, description → 指定gradio介面的標題與描述

'''
iface = gr.Interface(fn=answer_question, inputs="text", outputs="text", title="Interactive PDF", 
                     description="Enter your question in Chinese.")
iface.launch()  #Gradio，啟動!


