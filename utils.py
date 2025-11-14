import os
import json
import shutil
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())
chroma_db_path = "./chroma_db"

def load_documents_from_uploaded_pdf(uploaded_file) -> list[Document]:
    """Load documents from a Streamlit uploaded PDF file or file path.
    
    Args:
        uploaded_file: Either a Streamlit UploadedFile object or a string file path
        
    Returns:
        list[Document]: List of LangChain document objects
    """
    import tempfile
    
    if isinstance(uploaded_file, str):
        # If it's a file path, use it directly
        return load_documents_with_PyPDFLoader(uploaded_file)
    
    # For Streamlit uploaded file, we need to save it temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # If it's a Streamlit UploadedFile, it has a getvalue() method
            if hasattr(uploaded_file, 'getvalue'):
                tmp_file.write(uploaded_file.getvalue())
            # If it's a regular file-like object, use read()
            else:
                tmp_file.write(uploaded_file.read())
            tmp_file.flush()
            
            # Load documents using the temporary file
            documents = load_documents_with_PyPDFLoader(tmp_file.name)
            
        # Clean up the temporary file
        import os
        os.unlink(tmp_file.name)
        
        return documents
    except Exception as e:
        raise ValueError(f"Error loading PDF: {str(e)}")


def load_documents_with_PyPDFLoader(file_path: str) -> list[Document]:
    """Load documents from a PDF file using PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents: list[Document]) -> list[Document]:
    """Create chunks from documents using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
        length_function=len
    )
    all_splits  = text_splitter.split_documents(documents)
    return all_splits

def create_chroma_vector_store(documents: list[Document], embeddings) -> Chroma:
    """Create a Chroma vector store from documents or add to existing one.
    
    If a vector store already exists at chroma_db_path, adds the new documents to it.
    If no vector store exists, creates a new one.
    
    Args:
        documents: List of documents to add to the vector store
        embeddings: Embeddings model to use for vectorization
        
    Returns:
        Chroma: The vector store instance
    """
    # Check if vector store already exists
    if os.path.exists(chroma_db_path):
        # Load existing vector store
        vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings
        )
        # Add new documents to existing store
        vector_store.add_documents(documents)
    else:
        # Create new vector store if none exists
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_db_path
        )
    
    # Make sure to persist the changes
    #vector_store.persist()
    return vector_store


def print_vectore_store_info(vector_store):
    all_docs = vector_store.get()
    for key, value in all_docs.items():
        print(f'{key}: {value}')

def delete_vector_store():
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)

def create_rag_chain(llm, retriever):
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful analyst. 
        Use the following context to answer the question.
        
        Contexts may include financial reports, market analysis, table, and economic indicators.
        Respond in markdown format.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
                                                     
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
