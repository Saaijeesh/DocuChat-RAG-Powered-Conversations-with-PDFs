import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

##load the Groq API Key
groq_api_key=os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings(model="mistral")
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)
document_chain=create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)

prompt=st.text_input("Input your prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # find relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-------------------------------------")

# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# import time

# # Load environment variables from .env file
# load_dotenv()

# # Load the Groq API Key
# groq_api_key = os.getenv('GROQ_API_KEY')
# if not groq_api_key:
#     st.error("GROQ_API_KEY not found in environment variables.")
#     st.stop()

# # Initialize session state variables if not already present
# if "vector" not in st.session_state:
#     st.session_state.embeddings = OllamaEmbeddings(model="mistral")
#     st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
#     st.session_state.docs = st.session_state.loader.load()

#     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
#     st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# # Title of the Streamlit app
# st.title("ChatGroq Demo")

# # Initialize the language model with the API key and model name
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

# # Define the chat prompt template
# prompt_template = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question.
#     <context>
#     {context}
#     <context>
#     Questions: {input}
#     """
# )

# # Create the document chain
# document_chain = create_stuff_documents_chain(llm, prompt_template)

# # Retrieve the vector retriever from session state
# retriever = st.session_state.vector.as_retriever()

# # Create the retrieval chain
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # Input prompt from the user
# prompt = st.text_input("Input your prompt here")

# # Process the prompt if provided
# if prompt:
#     start = time.process_time()
#     response = retrieval_chain.invoke({"input": prompt})
#     st.write("Response time:", time.process_time() - start)
#     st.write(response['answer'])

#     # With a Streamlit expander, display document similarity search results
#     with st.expander("Document Similarity Search"):
#         # Find relevant chunks
#         for i, doc in enumerate(response['context']):
#             st.write(doc.page_content)
#             st.write("-------------------------------------")

