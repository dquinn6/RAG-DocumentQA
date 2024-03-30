import os
import logging
from logging.handlers import RotatingFileHandler
from config import config
import streamlit as st
import json
from src.communicators import GPTCommunicator
from src.data_processors import WikiTextProcessor
from src.vectorstore_handlers import LangchainVectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import sys

LOG_PATH = "logs/"
PATTERNS_FILENAME = 'config/manipulate_patterns.json'

# Direct src code logging to backend.log through logging.config
 
logging.basicConfig(
    filename=LOG_PATH+"backend.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

# Direct UI logging to streamlit.log through logger object

def create_logger(
    name: str = "logger_",
    level: str = "INFO",
    filename: str = LOG_PATH+"streamlit.log",
    max_log_size: int = 1024 * 1024 * 1024, #100 MB
    backup_count: int = 1,
):
    """ Create logger for handling streamlit IO

    Solution to duplicate log messages; found at https://discuss.streamlit.io/t/streamlit-duplicates-log-messages-when-stream-handler-is-added/16426

    Args:
        name (str, optional): _description_. Defaults to "logger_".
        level (str, optional): _description_. Defaults to "DEBUG".
        filename (str, optional): _description_. Defaults to "app.log".
        max_log_size (int, optional): _description_. Defaults to 1024*1024*1024.

    Returns:
        _type_: _description_
    """
    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    # if no handler present, add one
    if(
        sum([isinstance(handler, RotatingFileHandler) for handler in logger.handlers]) == 0
    ):
        handler = RotatingFileHandler(
            filename, maxBytes=max_log_size, backupCount=backup_count,
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
    
    return logger

def update_patterns_json(key = "", val = "", clear_json=False):

    with open(PATTERNS_FILENAME) as r:
        patterns_json = json.load(r)

    if clear_json:
        patterns_json = {}
    else:
        patterns_json.update({key: val})

    with open(PATTERNS_FILENAME, "w") as w:
        json.dump(patterns_json, w)

@st.cache_data
def init_session():
    # clear logs
    with open(LOG_PATH+"streamlit.log", "w") as _:
        pass
    with open(LOG_PATH+"backend.log", "w") as _:
        pass
    #clear patterns file
    update_patterns_json(clear_json=True)

@st.cache_resource
def init_communicator_and_processor():

    api_key = config.user_config["ACCESS_TOKEN"]
    model_name = config.user_config["MODEL_NAME"]
    verbose = config.user_config["VERBOSE"]

    communicator = GPTCommunicator(api_key=api_key, model_name=model_name)

    data_processor = WikiTextProcessor(
        dataset_version = "wikitext-2-raw-v1", 
        split = "train", 
        communicator = communicator,
        verbose = verbose
    )
    return communicator, data_processor

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

def run_streamlit_app():

    st.set_page_config(page_title="Page")

    if "logger" not in st.session_state:
        st.session_state["logger"] = create_logger()
    logger = st.session_state["logger"]

    init_session()

    communicator, data_processor = init_communicator_and_processor()

    st.title("Title")
    
    st.header("Initialization")

    # Document manipulation through Streamlit UI

    st.subheader("Document Manipulation")

    st.sidebar.header("Manipulate Documents")

    search_pattern_col, replace_pattern_col = st.sidebar.columns(2)
    search_pattern = search_pattern_col.text_input("Search Pattern")
    replace_pattern = replace_pattern_col.text_input("Replace Pattern")

    if st.sidebar.button("Add Pattern"):
        update_patterns_json(key=search_pattern, val=replace_pattern)

    if st.sidebar.button("Clear Patterns"):
        update_patterns_json(clear_json=True)

    with open(PATTERNS_FILENAME) as r:
        patterns_json = json.load(r)

    data = {"search": patterns_json.keys(), "replace": patterns_json.values()}

    st.write(pd.DataFrame(data))

    # Adjust params

    st.subheader("Changed Config Params")
    
    st.sidebar.header("Adjust Config")
    token_limit = st.sidebar.text_input("Token Limit")
    
    if token_limit:
        token_limit = int(token_limit)
        config.user_config["TOKEN_LIMIT"] = token_limit
        st.write(f"Token limit: {token_limit}")

    # Vectorstore init with manipulated docs

    st.header("Chatbot QA")

    load_vs, create_vs = st.columns(2)
    # load_vs = st.button("Load Vectorstore")
    # create_vs = st.button("Create New Vectorstore")

    if load_vs.button("Load vectorstore"):
        process_data(data_processor)
        communicator = attach_vectorstore(communicator, load_vectorstore=True)
        st.session_state["communicator"] = communicator

    elif create_vs.button("Create new vectorstore"):
        process_data(data_processor)
        result_holder = st.empty()

        def progress(p, i):
            with result_holder.container():
                st.progress(p, f'Progress: Documents Processed={i}')

        communicator = attach_vectorstore(communicator, load_vectorstore=False, _callback=progress)
        st.session_state["communicator"] = communicator

    else:
        if "communicator" not in st.session_state:
            st.session_state["communicator"] = None
    
    if st.session_state["communicator"] != None:

        #n_patterns = st.sidebar.slider("Manipulate Patterns:", min_value=1, max_value=10)

        user_query = st.text_input("Query: ")

        if st.button("Get Answer"):
            if user_query:

                # do rag
                response, retrieved_context = st.session_state["communicator"].post_rag_prompt(user_query)

                st.subheader("Response:")
                st.write(response)

                logger.info(f"User query: {user_query}")

                # Get user feedback

                feedback = st.radio(
                    "Select an option:",
                    [
                        "None",
                        "Yes",
                        "No",
                    ],
                )
                if feedback != "None":
                    logging.info(feedback)

    st.sidebar.header("Session logs")
    with open(LOG_PATH+"streamlit.log") as log:
        st.sidebar.write(log.readlines())     

    st.sidebar.header("Developer logs")
    with open(LOG_PATH+"backend.log") as log:
        st.sidebar.write(log.readlines())

run_streamlit_app()
