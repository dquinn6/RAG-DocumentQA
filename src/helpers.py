from config import config
import json
from src.vectorstore_handlers import LangchainVectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings

LOG_PATH = config.user_config["LOG_PATH"]
PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]

def update_patterns_json(key = None, val = None, clear_json=False):

    with open(PATTERNS_FILENAME) as r:
        patterns_json = json.load(r)

    if clear_json:
        patterns_json = {}

    else:
        if ((key not in [None, ""]) and (val not in [None, ""])):
            patterns_json.update({key: val})

    with open(PATTERNS_FILENAME, "w") as w:
        json.dump(patterns_json, w)

def process_data(data_processor):

    save_path = config.user_config["SAVE_PATH"]
    token_limit = config.user_config["TOKEN_LIMIT"]

    with open(PATTERNS_FILENAME) as f:
        patterns = list(json.load(f).items())
    
    _ = data_processor.process_text(
        token_limit = token_limit, 
        save_path = save_path,
        save_filename = "processed_data.csv",
        manipulate_pattern = patterns
    )

def attach_vectorstore(communicator, load_vectorstore=True, _callback=None):

    save_path = config.user_config["SAVE_PATH"]
    search_type = config.user_config["SEARCH_TYPE"]
    n_retrieved_docs = config.user_config["N_RETRIEVED_DOCS"]
    verbose = config.user_config["VERBOSE"]

    vs = LangchainVectorstore(
        embedding_type = HuggingFaceEmbeddings(),
        processed_csv_path = save_path+"processed_data.csv",
        verbose = verbose
    )

    if load_vectorstore:
        vs.load_local_vectorstore(load_path=save_path)
    else:
        vs.create_local_vectorstore(save_path=save_path, force_create=True, callback=_callback)

    vs.create_retriever(
        search_type=search_type,
        search_kwargs={
            "k": n_retrieved_docs
        }
    )

    communicator.set_vectorstore_handler(vs)

    return communicator