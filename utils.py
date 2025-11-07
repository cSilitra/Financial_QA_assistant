import os
import json
import shutil
from dotenv import load_dotenv, find_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared,operations
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements, elements_to_json
from langchain.chat_models import init_chat_model
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from typing import Callable, TypedDict
_ = load_dotenv(find_dotenv())

#llm = ChatOpenAI(temperature = 0.0, model='gpt-5')


def load_documents_with_PyPDFLoader(file_path: str) -> list[Document]:
    """Load documents from a PDF file using PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents: list[Document]) -> list[Document]:
    """Create chunks from documents using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # chunk size (characters)
        chunk_overlap=20,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits  = text_splitter.split_documents(documents)
    return all_splits

def create_chroma_vector_store(documents: list[Document], embeddings, chroma_db_path) -> Chroma:
    """Create a Chroma vector store from documents."""
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_db_path  # Optional: to save locally
    )
    return vector_store


def load_pdf_with_unstructured(file_path):
    UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
    try:
        loader = UnstructuredLoader([file_path])
                                    #api_key=UNSTRUCTURED_API_KEY, 
                                    #partition_via_api=True)
        return loader.load()
    except SDKError as e:
        print(e)
    

def preprocess_pdf_with_unstructured(file_path):
    UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
    client  = UnstructuredClient(
        api_key_auth=UNSTRUCTURED_API_KEY,
        #server_url=UNSTRUCTURED_API_URL,
    )
    
    
    # read the document through a unstructured API
    with open(file_path, "rb") as f:
        files=shared.Files(
            content=f.read(), 
            file_name=file_path,
        )

    req = shared.PartitionParameters(
        files=files,
        strategy='hi_res',
        #pdf_infer_table_structure=True,
        languages=["eng"],
    )
    try:
        request = operations.PartitionRequest(partition_parameters=req)
        resp = client.general.partition(request=request)

        return resp.elements
    except SDKError as e:
        print(e)

def get_openAI_model(model):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    model = init_chat_model(model,
            api_key=OPENAI_API_KEY,
            temperature=0.5,
            timeout=10,
            max_tokens=1000
            )
    
def get_claude_model(model):
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

    model = init_chat_model(model,
            api_key=CLAUDE_API_KEY,
            temperature=0.5,
            timeout=10,
            max_tokens=1000
            )
    
def print_vectore_store_info(vector_store):
    #all_docs = vector_store.get()['documents']
    all_docs = vector_store.get()
    for key, value in all_docs.items():
        print(f'{key}: {value}')

def delete_vector_store(chroma_db_path):
    #vector_store.delete_collection()
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)

def create_rag_chain(llm, retriever):
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant specializad in stock market. 
        Use the following context to answer the question.
        
        Contexts may include financial reports, market analysis, table, and economic indicators.
        Respond in markdown format.
    
        Context:
        {context}

        Question:
        {question}
        """)
    
    rag_chain = (
        RunnableMap({
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
    )
    return rag_chain
