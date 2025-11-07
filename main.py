import os
from utils import *
from IPython.display import Markdown, display
from rich.markdown import Markdown
from rich.console import Console
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

chroma_db_path = "./chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = get_openAI_model("gpt-5-mini")
file_path = "googleQ32025.pdf"

# 0. Init
llm = ChatOpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def create_vector_store():
    delete_vector_store(chroma_db_path)

    # 1. Load and preprocess the PDF document
    documents = load_documents_with_PyPDFLoader(file_path)

    # 2 Create chunks from the documents
    all_splits = create_chunks(documents)

    # 3. Create a vector store from the chunks
    vector_store = create_chroma_vector_store(all_splits, embeddings, chroma_db_path)

    retriever = vector_store.as_retriever()
    #return retriever

    # 5. Build the RAG pipeline using LCEL
    rag_chain =  create_rag_chain(llm, retriever)

    # 6. Run the RAG pipeline
    response = rag_chain.invoke("Which is the dividend policy of the company?")
    console = Console()
    console.print(Markdown(response.content))


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
    #if not os.path.exists(chroma_db_path):
        #create_vector_store()
    
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever()
    rag_chain =  create_rag_chain(llm, retriever)
    response = rag_chain.invoke(query)
    return response.content
    return response

create_vector_store()



