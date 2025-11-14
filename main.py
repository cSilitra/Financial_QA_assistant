import os
from utils import *
from IPython.display import Markdown, display
from rich.markdown import Markdown
from rich.console import Console
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from geminiFileSearchUtils import run_gemini_search_rag
from openai import OpenAI
    

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
file_path = "googleQ32025.pdf"

# 0. Init
llm = ChatOpenAI(api_key=OPENAI_API_KEY,model="gpt-5-mini")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def crate_vector_store(file):
    # 1. Load and preprocess the PDF document
    documents = load_documents_from_uploaded_pdf(file)

    # 2 Create chunks from the documents
    all_splits = create_chunks(documents)

    # 3. Create a vector store from the chunks
    vector_store = create_chroma_vector_store(all_splits, embeddings)


def init():
    delete_vector_store()

    # 1. Load and preprocess the PDF document
    documents = load_documents_with_PyPDFLoader(file_path)

    # 2 Create chunks from the documents
    all_splits = create_chunks(documents)

    # 3. Create a vector store from the chunks
    vector_store = create_chroma_vector_store(all_splits, embeddings)

    #retriever = vector_store.as_retriever()
    #return retriever

    # 5. Build the RAG pipeline using LCEL
    #rag_chain =  create_rag_chain(llm, retriever)

    # 6. Run the RAG pipeline
    #response = rag_chain.invoke("Which is the dividend policy of the company?")
    #console = Console()
    #console.print(Markdown(response.content))


def find_in_document():
    # Later or in another script, just load it
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever()
    #return retriever

    # 5. Build the RAG pipeline using LCEL
    rag_chain =  create_rag_chain(llm, retriever)

    # 6. Run the RAG pipeline
    response = rag_chain.invoke("Which is the dividend policy of the company?")
    return response.content
    
def run_master_agent(query: str):
    print("Running master agent...")
    
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever()
    rag_chain =  create_rag_chain(llm, retriever)
    response = rag_chain.invoke(query)
    print ('base',response)
    return response.content

def run_gemini_file_search(query: str):
    print("Running gemini search rag...")
    response = run_gemini_search_rag(query)
    print ('gemini',response)
    return response

def run_base_llm(query: str):
    print("Running gpt-5-mini...")
    openAiclient = OpenAI(api_key=OPENAI_API_KEY)
    response = openAiclient.responses.create(
    model="gpt-5-mini",
    input=query)
    return response.output_text

def run_app():
    question ="how much is the dividend when it will be paid, and for which Company’s Class"
    response = run_master_agent(question)
    print(response) # The response is already in markdown format from the RAG chain

#run_app()

