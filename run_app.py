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

    section1_color = "green" # Initializaion section
    section2_color = "blue" # Chatbot section
    sidebar1_color = "orange" # Adjust section
    sidebar2_color = "red" # Log section
    
    if "logger" not in st.session_state:
        st.session_state["logger"] = create_logger()
    logger = st.session_state["logger"]

    init_session()

    communicator, data_processor = init_communicator_and_processor()

    st.title("Title")
    
    st.header("Initialization", divider=section1_color)

    # Document manipulation through Streamlit UI

    st.sidebar.header("Manipulate Documents", divider=sidebar1_color)

    search_pattern_col, replace_pattern_col = st.sidebar.columns(2)
    search_pattern = search_pattern_col.text_input("Search Pattern")
    replace_pattern = replace_pattern_col.text_input("Replace Pattern")

    if "docs_with_pattern" not in st.session_state:
        st.session_state["docs_with_pattern"] = None

    if "enable_doc_viewer" not in st.session_state:
        st.session_state["enable_doc_viewer"] = False

    if st.sidebar.button("Add Pattern"):
        update_patterns_json(key=search_pattern, val=replace_pattern)
        docs_with_pattern = data_processor.ret_passages_with_pattern(search_pattern)
        st.sidebar.write(f"{len(docs_with_pattern)} / {len(data_processor.data)} documents with this search pattern.")
        st.session_state["docs_with_pattern"] = docs_with_pattern
        
    if st.session_state["docs_with_pattern"]:
        if st.sidebar.button("View these documents"):
            st.session_state["enable_doc_viewer"] = True

    if st.session_state["enable_doc_viewer"]:
        st.subheader("Doc Viewer", divider=section1_color)

        st.write(f"Search: {search_pattern}")

        docviewer = st.empty() 
        idx_placeholder = st.empty()

        # Initialize the current index
        if "current_index" not in st.session_state:
            st.session_state["current_index"] = 0

        show_next = st.button("next")

        # update index on button click
        if show_next:
            # loop back to beginning if at end of list
            if st.session_state["current_index"] == len(st.session_state["docs_with_pattern"]) - 1:
                st.session_state["current_index"] = 0
            else:
                st.session_state["current_index"] += 1

        with idx_placeholder.container(border=False):
            st.write(f"Document: {st.session_state.current_index + 1} / {len(st.session_state.docs_with_pattern)}")

        # Show next element in list
        with docviewer.container(height=300, border=True):
            st.write(st.session_state["docs_with_pattern"][st.session_state["current_index"]])

    if st.sidebar.button("Clear Patterns"):
        update_patterns_json(clear_json=True)
        st.session_state["docs_with_pattern"] = None

    with open(PATTERNS_FILENAME) as r:
        patterns_json = json.load(r)

    data = {"search": patterns_json.keys(), "replace": patterns_json.values()}

    st.subheader("Document Manipulation", divider=section1_color)

    st.write(pd.DataFrame(data))

    # Adjust params

    st.subheader("Config Params", divider=section1_color)

    # init with default config vals
    token_count_placeholder = st.empty()
    with token_count_placeholder.container(border=False):
        st.write(f"Token limit: {config.user_config['TOKEN_LIMIT']}")

    model_placeholder = st.empty()
    with model_placeholder.container(border=False):
        st.write(f"Model: {config.user_config['MODEL_NAME']}")

    st.sidebar.header("Adjust Config", divider=sidebar1_color)

    # Input to change model

    selected_model = st.sidebar.selectbox(
        'Select a model:',
        (
            "gpt-3.5-turbo", 
            "gpt-4", 
            "gpt-4-32k"
        )
    )
    if selected_model:
        config.user_config["MODEL_NAME"] = selected_model
        with model_placeholder.container(border=False):
            st.write(f"Model: {selected_model}")

    # Input to adjust token limit
    token_limit = st.sidebar.text_input("Token Limit")
    
    if token_limit:
        token_limit = int(token_limit)
        config.user_config["TOKEN_LIMIT"] = token_limit
        with token_count_placeholder.container(border=False):
            st.write(f"Token limit: {token_limit}")

    # Vectorstore init with manipulated docs

    st.header("Chatbot QA", divider=section2_color)

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

                show_context = st.button("Show retrieved docs in Doc Viewer")

                if show_context:
                    with docviewer.container(height=300, border=True):
                        st.write(retrieved_context)


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

    st.sidebar.header("Session logs", divider=sidebar2_color)
    with open(LOG_PATH+"streamlit.log") as log:
        st.sidebar.write(log.readlines())     

    st.sidebar.header("Developer logs", divider=sidebar2_color)
    with open(LOG_PATH+"backend.log") as log:
        st.sidebar.write(log.readlines())

run_streamlit_app()
