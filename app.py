import streamlit as st
from typing import List, Union, Optional # make type annotations in code

from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
from PyPDF2 import PdfReader

# Prompts for user-supplied information and question and answer format
PROMPT_TEMPLATE = """
Use the following pieces of context enclosed by triple backquotes to answer the question at the end.
\n\n
Context:
```
{context}

```
\n\n
Question: [][][][]{question}[][][][]  
\n
Answer:"""

# Initialize streamlit interface
def init_page() -> None:
    st.set_page_config(
        page_title="PDF Chat Assistant"

    )
    st.sidebar.title("Options")

# Initialize conversation message
def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content = (
                    "You are a helpful AI QA assistant."
                )
            )
        ]
        st.session_state.costs = []

# Extract text from uploaded files
def get_pdf_text() -> Optional[str]:
    st.header("File Update")
    uploaded_file = st.file_uploader(
        label = "Upload your PDF file here",
        type =  "pdf"
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file) 
        text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0) 
        return text_splitter.split_text(text)
    
    else:
        return None

# Construct vector store
def build_vector_store(
    texts: str, embeddings: Union[OpenAIEmbeddings, LlamaCppEmbeddings]) \
        -> Optional[Qdrant]:

        if texts:
            with st.spinner("Loading PDF..."):
                qdrant = Qdrant.from_texts(
                      texts,
                      embeddings,
                      path = ":memory:",
                      collection_name = "my_collection",
                      force_recreate = True    
                )
            st.success("File Loaded Sucessfully")
        
        else:
            qdrant = None
        return qdrant

# Provide users with a choice of LLM models
def select_llm() -> Union[ChatOpenAI, LlamaCpp]:
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-3.5-turbo-0613",
                                   "gpt-4",))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    return model_name, temperature

# Load the corresponding language model according to user selection
def load_llm(model_name: str, temperature: float) -> Union[ChatOpenAI, LlamaCpp]:
    if model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=temperature, model_name=model_name)

# Vector embedding(向量嵌入)
def load_embeddings(model_name: str) -> Union[OpenAIEmbeddings, LlamaCppEmbeddings]:
    if model_name.startswith("gpt-"):
        return OpenAIEmbeddings()
    
# Use the selected language model to answer the question   
def get_answer(llm, messages) -> tuple[str, float]:
    if isinstance(llm, ChatOpenAI):
        with get_openai_callback() as cb:
            answer = llm(messages)
        return answer.content, cb.total_cost
    
# Return the corresponding character according to the type of message   
def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    if isinstance(message, SystemMessage):
        return "System"
    
    if isinstance(message, HumanMessage):
        return "user"
    
    if isinstance(message, AIMessage):
        return "assistant"
    
    raise TypeError("Unknown message type.")

# Convert message to dictionary list
def convert_langchainchema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    
    return [{"role": find_role(message),
             "content": message.content} for message in messages]

# Extract questions from user-provided messages
def extract_userquestion_part_only(content):
    content_split = content.split("[][][][]") 
    if len(content_split) == 3:
        return content_split[1]
    return content

def main() -> None:
    _ = load_dotenv(find_dotenv())

    init_page()

    model_name, temperature = select_llm()
    llm = load_llm(model_name, temperature)
    embeddings = load_embeddings(model_name)

    texts = get_pdf_text()
    qdrant = build_vector_store(texts, embeddings)

    init_messages()

    st.header("(PDF) Chat Assistant")
    if user_input := st.chat_input("Enter your question here:"):
        if qdrant:
            context = [c.page_content for c in qdrant.similarity_search(user_input, k=10)]
            user_input_w_context = PromptTemplate(template=PROMPT_TEMPLATE, 
                                                  input_variables=["context", "question"]) \
                                                  .format(context=context, question=user_input)
            
        else:
            user_input_w_context = user_input
        st.session_state.messages.append(
            HumanMessage(content=user_input_w_context))
        with st.spinner("Bot is typing..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(extract_userquestion_part_only(message.content))

if __name__ == "__main__":
    main()