from unstructured_client import UnstructuredClient
from unstructured_client.models import shared,operations
from unstructured_client.models.errors import SDKError
from langchain_unstructured import UnstructuredLoader
from langchain.chat_models import init_chat_model

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